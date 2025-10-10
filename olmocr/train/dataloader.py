import base64
import hashlib
import json
import logging
import pickle
import re
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeAlias,
    Union,
    get_args,
    get_origin,
)

import numpy as np
import yaml
from PIL import Image
from pypdf import PdfReader
from torch.utils.data import Dataset
from tqdm import tqdm

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.anchor import get_anchor_text
from olmocr.prompts.prompts import PageResponse, build_finetuning_prompt

# Type alias for samples
Sample: TypeAlias = Dict[str, Any]

# Sentinel used for distinguishing cache misses
_CACHE_MISS = object()

# Configure logging
logger = logging.getLogger(__name__)


def validate_pdf_pair(md_path: Path) -> Tuple[Optional[Dict[str, Path]], Optional[Tuple[Path, str]]]:
    """Validate a single markdown-PDF pair.

    Args:
        md_path: Path to the markdown file

    Returns:
        Tuple of (valid_sample, invalid_pdf_info)
        - valid_sample: Dict with markdown_path and pdf_path if valid, None otherwise
        - invalid_pdf_info: Tuple of (pdf_path, reason) if invalid, None otherwise
    """
    # Look for PDF with same stem (filename without extension)
    pdf_path = md_path.with_suffix(".pdf")

    if pdf_path.exists() or pdf_path.is_symlink():
        # Resolve symlink if it is one
        if pdf_path.is_symlink():
            pdf_path = pdf_path.resolve()

        # Verify the resolved path exists
        if pdf_path.exists():
            # Validate PDF - check it loads and has exactly one page and that you can get document-anchoring from it
            try:
                reader = PdfReader(str(pdf_path))
                num_pages = len(reader.pages)

                if num_pages != 1:
                    return None, (pdf_path, f"Expected 1 page, found {num_pages}")

                # Test that document anchoring works
                from olmocr.prompts.anchor import get_anchor_text

                get_anchor_text(pdf_path, page=1, pdf_engine="pdfreport", target_length=100)

                return {"markdown_path": md_path, "pdf_path": pdf_path}, None

            except Exception as e:
                return None, (pdf_path, f"Failed to load: {str(e)}")

    return None, None


@dataclass(frozen=True, slots=True)
class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    @abstractmethod
    def __call__(self, sample: Sample) -> Optional[Sample]:
        """Process a sample and return the modified sample, or None to skip this sample."""
        ...

    def is_cacheable(self) -> bool:
        """Whether the step can be included in cached preprocessing."""
        return True

    def cache_fingerprint(self) -> str:
        """Return a deterministic fingerprint representing this step's configuration."""
        if not hasattr(self, "__dataclass_fields__"):
            return self.__class__.__name__

        parts = []
        for field_name in sorted(self.__dataclass_fields__.keys()):  # type: ignore[attr-defined]
            value = getattr(self, field_name)
            parts.append(f"{field_name}={self._normalize_cache_value(value)}")

        joined = ",".join(parts)
        return f"{self.__class__.__name__}({joined})"

    def _normalize_cache_value(self, value: Any) -> str:
        """Normalize values to stable string representations for caching."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return str(value)
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(self._normalize_cache_value(v) for v in value) + "]"
        if isinstance(value, dict):
            return "{" + ",".join(
                f"{k}:{self._normalize_cache_value(v)}" for k, v in sorted(value.items())
            ) + "}"
        if isinstance(value, type):
            return f"{value.__module__}.{value.__qualname__}"
        processor_name = getattr(value, "pretrained_model_name_or_path", None)
        if processor_name:
            return str(processor_name)
        tokenizer_name = getattr(value, "name_or_path", None)
        if tokenizer_name:
            return str(tokenizer_name)
        return value.__class__.__name__


class BaseMarkdownPDFDataset(Dataset):
    """Base dataset class that loads and verifies markdown-PDF pairs."""

    def __init__(
        self,
        root_dir: str | PathLike,
        pipeline_steps: Optional[List[PipelineStep]] = None,
        cache_dir: Optional[str | PathLike] = None,
    ):
        """
        Initialize the dataset by finding all markdown files with corresponding PDFs.

        Args:
            root_dir: Path to the root folder containing processed markdown and PDF files
            pipeline_steps: Optional list of pipeline steps to apply to each sample
            cache_dir: Optional directory for cached preprocessing results
        """
        self.root_dir = Path(root_dir)
        self.pipeline_steps = list(pipeline_steps or [])
        self.samples: List[Sample] = []

        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_enabled = self.cache_dir is not None
        self.cacheable_steps, self.runtime_steps = self._split_pipeline_steps()
        if self.cache_enabled and not self.cacheable_steps:
            logger.info("Cache requested for %s but no cacheable steps detected; disabling cache.", self.root_dir)
            self.cache_enabled = False
            self.cacheable_steps = list(self.pipeline_steps)
            self.runtime_steps = []
        if not self.cache_enabled:
            self.cacheable_steps = list(self.pipeline_steps)
            self.runtime_steps = []

        self.cache_identifier: Optional[str] = None
        self.cache_base_path: Optional[Path] = None
        self.cache_samples_dir: Optional[Path] = None
        self._cache_index_paths: List[Path] = []

        # Find all markdown files recursively
        logger.info(f"Scanning for markdown files in {self.root_dir}...")
        md_files = list(self.root_dir.rglob("*.md"))

        # Verify each markdown file has a corresponding PDF using ProcessPoolExecutor
        valid_count = 0
        invalid_pdfs = []

        logger.info(f"Validating {len(md_files)} markdown-PDF pairs using ProcessPoolExecutor...")

        # Use ProcessPoolExecutor for parallel validation
        with ProcessPoolExecutor(max_workers=8) as executor:
            # Submit all validation tasks
            future_to_md = {executor.submit(validate_pdf_pair, md_path): md_path for md_path in md_files}

            # Process results as they complete
            with tqdm(total=len(md_files), desc="Validating PDFs") as pbar:
                for future in as_completed(future_to_md):
                    md_path = future_to_md[future]
                    try:
                        valid_sample, invalid_pdf_info = future.result()

                        if valid_sample:
                            self.samples.append(valid_sample)
                            valid_count += 1
                        elif invalid_pdf_info:
                            invalid_pdfs.append(invalid_pdf_info)

                    except Exception as e:
                        logger.error(f"Error processing {md_path}: {str(e)}")
                        invalid_pdfs.append((md_path.with_suffix(".pdf"), f"Processing error: {str(e)}"))

                    pbar.update(1)

        # Sort samples by markdown path for consistent ordering across runs
        self.samples.sort(key=lambda x: x["markdown_path"])

        logger.info(f"Found {valid_count} valid markdown-PDF pairs")

        if invalid_pdfs:
            logger.warning(f"{len(invalid_pdfs)} invalid PDFs found:")
            for pdf_path, reason in invalid_pdfs[:5]:  # Show first 5
                logger.warning(f"  - {pdf_path.name}: {reason}")
            if len(invalid_pdfs) > 5:
                logger.warning(f"  ... and {len(invalid_pdfs) - 5} more")

        self._initialize_cache()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a single sample from the dataset.

        Returns:
            dict containing at minimum:
                - 'markdown_path': Path to the markdown file
                - 'pdf_path': Path to the PDF file

            Additional fields will be added by pipeline steps.
            Returns None if any pipeline step returns None.
        """
        base_sample = self.samples[idx].copy()

        if self.cache_enabled and self.cache_samples_dir is not None:
            cached_sample = self._load_cached_sample(idx)
            if cached_sample is _CACHE_MISS:
                cached_sample = self._apply_cacheable_steps(base_sample)
                self._store_cached_sample(idx, cached_sample)
            sample = cached_sample
        else:
            sample = self._apply_cacheable_steps(base_sample)

        if sample is None:
            return None

        for step in self.runtime_steps:
            sample = step(sample)
            if sample is None:
                return None

        return sample

    def _split_pipeline_steps(self) -> tuple[List[PipelineStep], List[PipelineStep]]:
        """Split pipeline steps into cacheable prefix and runtime suffix."""
        if not self.cache_enabled:
            return list(self.pipeline_steps), []

        cacheable: List[PipelineStep] = []
        runtime: List[PipelineStep] = []
        cache_prefix_active = True

        for step in self.pipeline_steps:
            if cache_prefix_active and step.is_cacheable():
                cacheable.append(step)
            else:
                cache_prefix_active = False
                runtime.append(step)

        if not cacheable:
            return list(self.pipeline_steps), []

        return cacheable, runtime

    def _initialize_cache(self) -> None:
        """Prepare cache directories and metadata if caching is enabled."""
        if not self.cache_enabled or not self.cacheable_steps:
            self.cache_identifier = None
            self.cache_base_path = None
            self.cache_samples_dir = None
            self._cache_index_paths = []
            return

        samples_hash = self._compute_samples_hash()
        cache_identifier = self._compute_cache_identifier(samples_hash)
        self.cache_identifier = cache_identifier
        self.cache_base_path = self.cache_dir / cache_identifier if self.cache_dir else None

        if self.cache_base_path is None:
            self.cache_enabled = False
            self.cache_samples_dir = None
            self._cache_index_paths = []
            return

        self.cache_base_path.mkdir(parents=True, exist_ok=True)
        self.cache_samples_dir = self.cache_base_path / "samples"
        self.cache_samples_dir.mkdir(parents=True, exist_ok=True)
        self._cache_index_paths = [self.cache_samples_dir / f"{idx:08d}.pkl" for idx in range(len(self.samples))]

        metadata = {
            "schema_version": 1,
            "samples_hash": samples_hash,
            "cacheable_steps": [step.cache_fingerprint() for step in self.cacheable_steps],
        }
        metadata_path = self.cache_base_path / "metadata.json"
        try:
            metadata_path.write_text(json.dumps(metadata, indent=2))
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning("Failed to write cache metadata to %s: %s", metadata_path, exc)

        logger.info(
            "Cache enabled for %s with %d samples; prefix steps: %s; cache dir: %s",
            self.root_dir,
            len(self.samples),
            [step.__class__.__name__ for step in self.cacheable_steps],
            self.cache_base_path,
        )

    def _compute_samples_hash(self) -> str:
        """Build a hash capturing dataset contents to invalidate stale caches."""
        hasher = hashlib.md5()

        for sample in self.samples:
            md_path = sample.get("markdown_path")
            pdf_path = sample.get("pdf_path")

            if md_path is not None:
                resolved_md = Path(md_path).resolve()
                hasher.update(str(resolved_md).encode("utf-8"))
                try:
                    stat = resolved_md.stat()
                    hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
                    hasher.update(str(stat.st_size).encode("utf-8"))
                except (OSError, FileNotFoundError):
                    pass

            if pdf_path is not None:
                resolved_pdf = Path(pdf_path).resolve()
                hasher.update(str(resolved_pdf).encode("utf-8"))
                try:
                    stat = resolved_pdf.stat()
                    hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
                    hasher.update(str(stat.st_size).encode("utf-8"))
                except (OSError, FileNotFoundError):
                    pass

        return hasher.hexdigest()

    def _compute_cache_identifier(self, samples_hash: str) -> str:
        """Derive a stable cache namespace from dataset configuration."""
        step_fingerprint = "||".join(step.cache_fingerprint() for step in self.cacheable_steps)
        root_signature = str(self.root_dir.resolve())
        digest_source = f"{root_signature}||{samples_hash}||{step_fingerprint}".encode("utf-8")
        digest = hashlib.md5(digest_source).hexdigest()
        safe_root = re.sub(r"[^A-Za-z0-9_.-]", "_", self.root_dir.name)
        return f"{safe_root}-{digest}"

    def _apply_cacheable_steps(self, sample: Sample) -> Optional[Sample]:
        """Apply cacheable steps sequentially."""
        for step in self.cacheable_steps:
            sample = step(sample)
            if sample is None:
                return None
        return sample

    def _load_cached_sample(self, idx: int) -> Any:
        """Load cached preprocessing result for the given index."""
        cache_file = self._cache_index_paths[idx]
        if not cache_file.exists():
            return _CACHE_MISS

        try:
            with cache_file.open("rb") as handle:
                return pickle.load(handle)
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning("Failed to read cache file %s: %s; regenerating entry.", cache_file, exc)
            try:
                cache_file.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            return _CACHE_MISS

    def _store_cached_sample(self, idx: int, sample: Optional[Sample]) -> None:
        """Persist preprocessing result for the given index."""
        cache_file = self._cache_index_paths[idx]
        tmp_file = cache_file.parent / f"{cache_file.name}.tmp"

        try:
            with tmp_file.open("wb") as handle:
                pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_file.replace(cache_file)
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning("Failed to write cache file %s: %s", cache_file, exc)
            try:
                tmp_file.unlink(missing_ok=True)
            except FileNotFoundError:
                pass


@dataclass(frozen=True, slots=True)
class FrontMatterParser(PipelineStep):
    """Pipeline step that parses YAML front matter from markdown content."""

    front_matter_class: Optional[Type] = None

    def _is_optional_str(self, field_type: Type) -> bool:
        """Check if a type is Optional[str]."""
        origin = get_origin(field_type)
        args = get_args(field_type)
        return origin is Union and type(None) in args and str in args

    def _extract_front_matter_and_text(self, markdown_content: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML front matter and text from markdown content."""
        if markdown_content.startswith("---\n"):
            try:
                # Find the closing --- delimiter
                end_index = markdown_content.find("\n---", 4)
                if end_index != -1:
                    front_matter_str = markdown_content[4:end_index]
                    text = markdown_content[end_index + 4 :].strip()

                    # Parse YAML
                    front_matter = yaml.safe_load(front_matter_str) or {}
                    return front_matter, text
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse YAML front matter: {e}")

        return {}, markdown_content.strip()

    def _parse_front_matter(self, front_matter_dict: Dict[str, Any], text: str) -> Any:
        """Parse front matter dictionary into dataclass instance if front_matter_class is specified."""
        if not self.front_matter_class:
            return front_matter_dict

        # Get field names and types from the dataclass
        field_info = {f.name: f.type for f in fields(self.front_matter_class)}

        # Validate and convert values
        kwargs = {}
        for field_name, field_type in field_info.items():
            # Special handling for natural_text field in PageResponse
            if field_name == "natural_text" and self.front_matter_class == PageResponse:
                kwargs[field_name] = text if text else None
                continue

            if field_name not in front_matter_dict:
                raise ValueError(f"Missing required field '{field_name}' in front matter")

            value = front_matter_dict[field_name]

            # Handle type conversions
            if field_type is int and isinstance(value, str):
                kwargs[field_name] = int(value)
            elif field_type is bool and isinstance(value, str):
                kwargs[field_name] = value.lower() == "true"
            elif self._is_optional_str(field_type):
                # Handle boolean values that YAML might produce (e.g., 'no' -> False)
                if isinstance(value, bool):
                    kwargs[field_name] = None
                elif isinstance(value, str):
                    kwargs[field_name] = value if value else None
                else:
                    kwargs[field_name] = None if not value else value
            else:
                kwargs[field_name] = value

        # Check for extra fields (excluding natural_text if it's PageResponse)
        expected_fields = set(field_info.keys())
        if self.front_matter_class == PageResponse:
            expected_fields.discard("natural_text")
        extra_fields = set(front_matter_dict.keys()) - expected_fields
        if extra_fields:
            raise ValueError(f"Unexpected fields in front matter: {extra_fields}")

        return self.front_matter_class(**kwargs)

    def __call__(self, sample: Sample) -> Sample:
        """Parse front matter from markdown content."""
        # Read markdown content if not already loaded
        if "markdown_content" not in sample:
            sample["markdown_content"] = sample["markdown_path"].read_text(encoding="utf-8")

        # Extract and parse front matter
        front_matter, text = self._extract_front_matter_and_text(sample["markdown_content"])

        # Parse front matter to dataclass if specified
        try:
            page_data = self._parse_front_matter(front_matter, text)
        except Exception as e:
            raise ValueError(f"Error parsing front matter for {sample['markdown_path']}: {e}")

        # Only add page_data field
        sample["page_data"] = page_data

        return sample


@dataclass(frozen=True, slots=True)
class PDFRenderer(PipelineStep):
    """Pipeline step that renders PDF to image."""

    target_longest_image_dim: int

    def __call__(self, sample: Sample) -> Sample:
        """Render PDF to image."""
        # Render PDF to image
        base64_png = render_pdf_to_base64png(str(sample["pdf_path"]), page_num=1, target_longest_image_dim=self.target_longest_image_dim)
        png_bytes = base64.b64decode(base64_png)
        image = Image.open(BytesIO(png_bytes))

        # Update sample
        sample["image"] = image

        return sample


@dataclass(frozen=True, slots=True)
class StaticLengthDocumentAnchoring(PipelineStep):
    target_anchor_text_len: int

    """Pipeline step that runs document anchoring on the PDF and puts in the data to be used by later prompting stages"""

    def __call__(self, sample: Sample) -> Sample:
        anchor_text = get_anchor_text(sample["pdf_path"], page=1, pdf_engine="pdfreport", target_length=self.target_anchor_text_len)
        sample["anchor_text"] = anchor_text
        return sample


@dataclass(frozen=True, slots=True)
class FinetuningPrompt(PipelineStep):
    """Applies the standard fine tuning prompt"""

    def __call__(self, sample: Sample) -> Sample:
        sample["instruction_prompt"] = build_finetuning_prompt(sample["anchor_text"])
        return sample


@dataclass(frozen=True, slots=True)
class NewYamlFinetuningPromptWithAnchoring(PipelineStep):
    """Applies the standard fine tuning prompt"""

    def __call__(self, sample: Sample) -> Sample:
        sample["instruction_prompt"] = (
            f"Attached is one page of a document, as well as some raw textual content that was previously extracted for it. "
            f"Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to markdown.\n"
            f"RAW_TEXT_START\n{sample['anchor_text']}\nRAW_TEXT_END\n"
            f"Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
        )
        return sample


@dataclass(frozen=True, slots=True)
class NewYamlFinetuningPromptWithNoAnchoring(PipelineStep):
    """Applies the standard fine tuning prompt"""

    def __call__(self, sample: Sample) -> Sample:
        sample["instruction_prompt"] = (
            f"Attached is one page of a document that you must process. "
            f"Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to markdown.\n"
            f"Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
        )
        return sample


@dataclass(frozen=True, slots=True)
class FrontMatterOutputFormat(PipelineStep):
    """Takes the output and applies the standard yaml formatting to it"""

    def __call__(self, sample: Sample) -> Sample:
        page_data = sample["page_data"]
        assert type(page_data) is PageResponse

        sample["response"] = (
            f"""---
primary_language: {page_data.primary_language}
is_rotation_valid: {page_data.is_rotation_valid}
rotation_correction: {page_data.rotation_correction}
is_table: {page_data.is_table}
is_diagram: {page_data.is_diagram}
---
{page_data.natural_text if page_data.natural_text is not None and len(page_data.natural_text.strip()) > 0 else ""}
""".strip()
        )

        return sample


@dataclass(frozen=True, slots=True)
class JSONOutputFormat(PipelineStep):
    """Takes the output and applies the standard yaml formatting to it"""

    def __call__(self, sample: Sample) -> Sample:
        page_data = sample["page_data"]
        assert type(page_data) is PageResponse

        sample["response"] = json.dumps(
            {
                "primary_language": page_data.primary_language,
                "is_rotation_valid": page_data.is_rotation_valid,
                "rotation_correction": page_data.rotation_correction,
                "is_table": page_data.is_table,
                "is_diagram": page_data.is_diagram,
                "natural_text": page_data.natural_text,
            },
            ensure_ascii=False,
        )

        return sample


@dataclass(frozen=True, slots=True)
class LatexBracketNormalizer(PipelineStep):
    """Normalizes LaTeX brackets in natural text field."""

    def __call__(self, sample: Sample) -> Sample:
        """Normalize LaTeX brackets in the natural text field."""
        # Get the page_data object
        if "page_data" not in sample:
            return sample

        page_data = sample["page_data"]
        if not hasattr(page_data, "natural_text") or not page_data.natural_text:
            return sample

        text = page_data.natural_text

        # Define patterns for LaTeX normalization
        # Order matters: process display math first, then inline
        patterns = [
            (r"\$\$(.+?)\$\$", r"\[\1\]"),  # $$...$$ to \[...\]
            (r"\$(.+?)\$", r"\(\1\)"),  # $...$ to \(...\)
        ]

        # Apply replacements
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.DOTALL)

        # Update the page_data with normalized text
        # Since PageResponse is frozen, we need to create a new instance
        from olmocr.prompts.prompts import PageResponse

        new_page_data = PageResponse(
            primary_language=page_data.primary_language,
            is_rotation_valid=page_data.is_rotation_valid,
            rotation_correction=page_data.rotation_correction,
            is_table=page_data.is_table,
            is_diagram=page_data.is_diagram,
            natural_text=text,
        )

        sample["page_data"] = new_page_data
        return sample


@dataclass(frozen=True, slots=True)
class RotationAugmentation(PipelineStep):
    """Pipeline step that randomly rotates images for augmentation."""

    probability: float = 0.5  # Probability of applying rotation

    def is_cacheable(self) -> bool:
        return False

    def __call__(self, sample: Sample) -> Optional[Sample]:
        """Randomly rotate image and update rotation metadata."""
        # Only proceed with given probability
        if np.random.random() > self.probability:
            return sample

        # Check if image exists
        if "image" not in sample:
            return sample

        # Check if page_data exists (we need to update it)
        if "page_data" not in sample:
            return sample

        # Randomly choose a rotation (90, 180, or 270 degrees)
        rotation_degrees = np.random.choice([90, 180, 270])

        # Apply rotation to image
        image = sample["image"]
        if rotation_degrees == 90:
            transpose = Image.Transpose.ROTATE_90
        elif rotation_degrees == 180:
            transpose = Image.Transpose.ROTATE_180
        else:  # 270
            transpose = Image.Transpose.ROTATE_270

        rotated_image = image.transpose(transpose)
        sample["image"] = rotated_image

        # Update page_data
        page_data = sample["page_data"]

        # Create new PageResponse with updated rotation info
        # The rotation_correction should be the inverse of what we applied
        # If we rotated 90 clockwise, we need 270 counter-clockwise to correct it
        if rotation_degrees == 90:
            correction = 270
        elif rotation_degrees == 180:
            correction = 180
        else:  # 270
            correction = 90

        from olmocr.prompts.prompts import PageResponse

        new_page_data = PageResponse(
            primary_language=page_data.primary_language,
            is_rotation_valid=False,  # Mark as invalid since we rotated it
            rotation_correction=correction,  # The correction needed to fix it
            is_table=page_data.is_table,
            is_diagram=page_data.is_diagram,
            natural_text=page_data.natural_text,
        )

        sample["page_data"] = new_page_data
        return sample


@dataclass(frozen=True, slots=True)
class FilterOutRotatedDocuments(PipelineStep):
    """Pipeline step that filters out documents with rotation issues."""

    def __call__(self, sample: Sample) -> Optional[Sample]:
        """Filter out samples where rotation is invalid or rotation correction is needed."""
        # Check if page_data exists
        if "page_data" not in sample:
            return sample

        page_data = sample["page_data"]

        # Check if page_data has the required attributes
        if not hasattr(page_data, "is_rotation_valid") or not hasattr(page_data, "rotation_correction"):
            return sample

        # Filter out if rotation is invalid or rotation correction is not 0
        if page_data.is_rotation_valid is False or page_data.rotation_correction != 0:
            return None

        return sample


@dataclass(frozen=True, slots=True)
class AugraphyBasicAugmentations(PipelineStep):
    """Pipeline step that applies a decent selection of augraphy augmentations to the data"""

    probability: float = 0.5  # Overall probability of applying any augmentation

    def is_cacheable(self) -> bool:
        return False

    def __call__(self, sample: Sample) -> Optional[Sample]:
        """Apply augraphy augmentations to the image in the sample."""
        # Check that the image data exists
        if "image" not in sample:
            return sample

        # Import opencv only here
        import cv2

        image = sample["image"]

        # Skip all augmentations based on overall probability
        if np.random.random() > self.probability:
            return sample

        # Convert from PIL to BGR for OpenCV/Augraphy
        image_numpy = np.array(image)
        if len(image_numpy.shape) < 3:
            image_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)

        # Apply a basic augraphy pipeline
        from augraphy import (
            AugraphyPipeline,
            Brightness,
            InkBleed,
            InkMottling,
            InkShifter,
            Jpeg,
            LowInkPeriodicLines,
            LowInkRandomLines,
            OneOf,
        )

        # Apply geometric transformations first, maintaing scale
        if np.random.random() < 0.50:
            # Get dimensions
            height, width = image_bgr.shape[:2]

            # Random parameters for geometric transformations
            angle = max(min(np.random.standard_normal(), 3), -3)  # Small rotation range
            scale = np.random.uniform(0.95, 1.05)  # Small scale range
            tx = np.random.uniform(-0.02, 0.02) * width  # Translation as fraction of width
            ty = np.random.uniform(-0.02, 0.02) * height  # Translation as fraction of height

            # Calculate center point
            center = (width / 2, height / 2)

            # Create transformation matrix
            M = cv2.getRotationMatrix2D(center, angle, scale)

            # Add translation
            M[0, 2] += tx
            M[1, 2] += ty

            # Apply transformation
            image_bgr = cv2.warpAffine(
                image_bgr,
                M,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),  # White background for documents
            )

        ink_phase = [
            OneOf([InkBleed(p=1), LowInkRandomLines(p=1), LowInkPeriodicLines(p=1), InkMottling(p=1), InkShifter(p=1, text_shift_scale_range=(10, 15))], p=0.2),
        ]

        paper_phase = [OneOf([Brightness(p=0.2), Jpeg(p=1)])]

        post_phase = [
            # Empty on purpose or else augmentations are too strong
        ]

        augmentation_pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

        # Apply augmentations
        augmented_image_bgr = augmentation_pipeline(image_bgr)

        # Convert back to RGB and then to PIL format
        augmented_image_rgb = cv2.cvtColor(augmented_image_bgr, cv2.COLOR_BGR2RGB)
        augmented_image_pil = Image.fromarray(augmented_image_rgb)

        # Update the sample with the augmented image
        sample["image"] = augmented_image_pil

        # Double-check PIL image size matches original
        assert augmented_image_pil.size == image.size, f"PIL image size changed during augmentation: {image.size} -> {augmented_image_pil.size}"

        return sample


@dataclass(frozen=True, slots=True)
class InstructUserMessages(PipelineStep):
    """Creates instruction-following messages format for training."""

    prompt_first: bool = False

    def __call__(self, sample: Sample) -> Sample:
        # Prepare messages
        if self.prompt_first:
            messages = {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["instruction_prompt"]},
                    {"type": "image", "image": sample["image"]},
                ],
            }
        else:
            messages = {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["instruction_prompt"]},
                ],
            }

        sample["user_messages"] = messages

        return sample


@dataclass(frozen=True, slots=True)
class Tokenizer(PipelineStep):
    """Tokenizes messages and creates training labels with proper masking."""

    processor: Any  # The model processor (e.g., AutoProcessor)
    masking_index: int = -100
    end_of_message_token: str = "<|im_end|>"  # Configurable, defaults to Qwen format

    def __call__(self, sample: Sample) -> Sample:
        """Tokenize messages and create labels for training."""
        if np is None:
            raise ImportError("numpy is required for Tokenizer step")

        # Extract user message and response
        user_messages = sample["user_messages"]
        response = sample["response"]

        # Apply chat template to user message only with generation prompt
        # user_messages is a single dict, so wrap it in a list
        text = self.processor.apply_chat_template([user_messages], tokenize=False, add_generation_prompt=True)

        main_image = None
        for usg_msg in user_messages["content"]:
            if "image" in usg_msg:
                main_image = usg_msg["image"]
                break

        assert main_image is not None

        # Process inputs using processor
        inputs = self.processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="np",
        )

        # Get labels by tokenizing the output text
        labels = self.processor(text=[response], padding=True, return_tensors="np")

        # Append end-of-message token to the labels
        end_tokens = self.processor.tokenizer(self.end_of_message_token, add_special_tokens=False)["input_ids"]
        end_tokens = np.array(end_tokens, dtype=inputs.input_ids.dtype)

        # Handle the case where labels['input_ids'] is empty
        if labels["input_ids"].shape[1] == 0:
            labels_input_ids_0 = np.array([], dtype=inputs.input_ids.dtype)
        else:
            labels_input_ids_0 = labels["input_ids"][0].astype(inputs.input_ids.dtype)

        labels["input_ids"] = np.concatenate([labels_input_ids_0, end_tokens])
        labels["input_ids"] = np.expand_dims(labels["input_ids"], axis=0)

        # Concatenate input_ids and labels
        input_ids = np.concatenate([inputs.input_ids[0], labels.input_ids[0]], axis=0)

        # All columns will participate in attention fully
        attention_mask = np.ones_like(input_ids)

        # Create labels, masking the input portion with -100
        labels_full = np.full_like(input_ids, fill_value=self.masking_index)
        labels_full[len(inputs.input_ids[0]) :] = labels.input_ids[0]

        # Return as dict, including pixel_values
        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask
        sample["labels"] = labels_full
        sample["pixel_values"] = inputs.pixel_values

        if hasattr(inputs, "image_grid_thw"):
            sample["image_grid_thw"] = inputs.image_grid_thw[0]

        return sample

    def cache_fingerprint(self) -> str:
        processor_name = getattr(self.processor, "pretrained_model_name_or_path", None)
        if processor_name is None:
            processor_name = getattr(self.processor, "name_or_path", None)

        tokenizer = getattr(self.processor, "tokenizer", None)
        tokenizer_name = None
        if tokenizer is not None:
            tokenizer_name = getattr(tokenizer, "name_or_path", None)

        identifier = processor_name or tokenizer_name or self.processor.__class__.__name__
        return (
            f"Tokenizer(processor={identifier},masking_index={self.masking_index},"
            f"end_of_message_token={self.end_of_message_token})"
        )


@dataclass(frozen=True, slots=True)
class RandomTokenFlipper(PipelineStep):
    """Randomly flips tokens in the output (non-masked) portion and masks their labels."""

    valid_token_ids: List[int]  # List of valid token IDs to substitute with
    token_flip_rate: float = 1e-4
    masking_index: int = -100

    def is_cacheable(self) -> bool:
        return False

    def __call__(self, sample: Sample) -> Sample:
        """Randomly flip tokens in the non-masked portion of labels."""
        if "labels" not in sample or "input_ids" not in sample:
            return sample

        # Work with copies to avoid modifying original arrays
        labels = sample["labels"].copy()
        input_ids = sample["input_ids"].copy()

        # Find indices where labels are not masked (i.e., output tokens)
        non_masked_indices = np.where(labels != self.masking_index)[0]

        if len(non_masked_indices) == 0:
            return sample

        # For each non-masked token, independently decide whether to flip
        for idx in non_masked_indices:
            if np.random.random() < self.token_flip_rate:
                # Pick a random token from the valid tokens list
                random_token = np.random.choice(self.valid_token_ids)
                input_ids[idx] = random_token
                labels[idx] = self.masking_index

        # Update sample with modified arrays
        sample["input_ids"] = input_ids
        sample["labels"] = labels

        return sample


class MarkdownPDFDocumentDataset(BaseMarkdownPDFDataset):
    """Dataset that includes front matter parsing and PDF rendering by default."""

    def __init__(
        self,
        root_dir: str | PathLike,
        target_longest_image_dim: int,
        front_matter_class=None,
        cache_dir: Optional[str | PathLike] = None,
    ):
        """
        Initialize the dataset with default pipeline steps.

        Args:
            root_dir: Path to the root folder containing processed markdown and PDF files
            target_longest_image_dim: Target dimension for the longest side of the image
            front_matter_class: Optional dataclass type to validate front matter against
        """
        # Create default pipeline steps
        pipeline_steps = [
            FrontMatterParser(front_matter_class),
            PDFRenderer(target_longest_image_dim),
            StaticLengthDocumentAnchoring(target_anchor_text_len=6000),
            FinetuningPrompt(),
            FrontMatterOutputFormat(),
            InstructUserMessages(),
        ]

        # Initialize base class with pipeline
        super().__init__(root_dir, pipeline_steps, cache_dir=cache_dir)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # Set up logging for testing
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Test MarkdownPDFDocumentDataset with YAML configuration")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="Which dataset subset to display (train or eval)",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="Index of dataset to use from the train/eval list",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of sample to display in detail",
    )
    parser.add_argument(
        "--analyze-tokens",
        action="store_true",
        help="Analyze token length distribution across entire dataset",
    )
    parser.add_argument(
        "--save-image",
        type=str,
        help="Save the processed image to the specified file path (e.g., output.png)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Optional cache directory for dataset preprocessing",
    )

    args = parser.parse_args()

    # Import config module
    from olmocr.train.config import Config

    # Load configuration
    print(f"\n=== Loading configuration from {args.config} ===")
    config = Config.from_yaml(args.config)

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        exit(1)

    # Load processor for tokenization
    print(f"\nLoading processor: {config.model.name}")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(config.model.name)

    # Select dataset based on type
    if args.dataset_type == "train":
        dataset_configs = config.dataset.train
        dataset_name = "train"
    else:
        dataset_configs = config.dataset.eval
        dataset_name = "eval"

    if args.dataset_index >= len(dataset_configs):
        print(f"Error: Dataset index {args.dataset_index} out of range. Only {len(dataset_configs)} {dataset_name} datasets available.")
        exit(1)

    dataset_cfg = dataset_configs[args.dataset_index]
    root_dir = dataset_cfg["root_dir"]
    pipeline_steps = config.get_pipeline_steps(dataset_cfg["pipeline"], processor)

    print(f"\n=== Testing {dataset_name} dataset {args.dataset_index} ===")
    print(f"Root directory: {root_dir}")
    print(f"Pipeline steps: {[step.__class__.__name__ for step in pipeline_steps]}")

    # Create dataset
    dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps, cache_dir=args.cache_dir)

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        # Show first few samples
        print("\nFirst 5 samples:")
        for i in range(min(5, len(dataset))):
            sample = dataset.samples[i]
            print(f"  {i}: MD: {sample['markdown_path'].name}, PDF: {sample['pdf_path'].name}")

        # Check if sample index is valid
        if args.sample_index >= len(dataset):
            print(f"\nError: Sample index {args.sample_index} out of range. Only {len(dataset)} samples available.")
            exit(1)

        # Get the requested sample
        print(f"\n=== Displaying sample {args.sample_index} ===")
        sample = dataset[args.sample_index]

        # Display sample information based on pipeline output
        print("\nSample keys:", list(sample.keys()))

        # If it's raw data (no tokenization)
        if "markdown_path" in sample:
            print(f"\nMarkdown file: {sample['markdown_path'].name}")
        if "pdf_path" in sample:
            print(f"PDF file: {sample['pdf_path'].name}")
        if "image" in sample and hasattr(sample["image"], "size"):
            print(f"Image size: {sample['image'].size}")

            # Save image if requested
            if args.save_image:
                sample["image"].save(args.save_image)
                print(f"Saved image to: {args.save_image}")

        if "page_data" in sample:
            print(f"\nPage data: {sample['page_data']}")
        if "messages" in sample:
            print(f"\n=== Messages ===")
            for i, msg in enumerate(sample["messages"]):
                print(f"\nMessage {i}:")
                print(f"  Role: {msg['role']}")
                print(f"  Content preview: {str(msg['content'])[:200]}...")

        # If it's tokenized data
        if "input_ids" in sample:
            print(f"\n=== Tokenized Output ===")
            print(f"  Keys: {list(sample.keys())}")
            print(f"  Input IDs shape: {sample['input_ids'].shape}")
            print(f"  Labels shape: {sample['labels'].shape}")
            print(f"  Attention mask shape: {sample['attention_mask'].shape}")

            if "pixel_values" in sample:
                print(f"  Pixel values shape: {sample['pixel_values'].shape}")
            if "image_grid_thw" in sample:
                print(f"  Image grid THW: {sample['image_grid_thw']}")

            # Show label masking
            print(f"\nLabel masking analysis:")
            labels = sample["labels"]
            masked_count = np.sum(labels == -100)
            total_count = len(labels)
            print(f"  Total tokens: {total_count}")
            print(f"  Masked tokens: {masked_count} ({masked_count/total_count*100:.1f}%)")
            print(f"  Unmasked tokens: {total_count - masked_count} ({(total_count - masked_count)/total_count*100:.1f}%)")

            # Find the transition point
            transition_idx = None
            for i in range(len(labels) - 1):
                if labels[i] == -100 and labels[i + 1] != -100:
                    transition_idx = i + 1
                    break

            if transition_idx:
                print(f"  Transition from masked to unmasked at position: {transition_idx}")

            # Print all tokens
            input_ids = sample["input_ids"]
            print(f"\nAll tokens ({len(input_ids)} total):")
            print("Format: [index] Token (repr) | Label | Token ID")
            print("-" * 80)

            for i in range(len(input_ids)):
                token = processor.tokenizer.decode([input_ids[i]])
                token_repr = repr(token)
                label = labels[i] if i < len(labels) else "N/A"
                token_id = input_ids[i]

                # Mark special positions
                marker = ""
                if transition_idx and i == transition_idx:
                    marker = " <-- TRANSITION (first unmasked)"
                elif i == 0:
                    marker = " <-- START"
                elif label != -100 and i > 0 and labels[i - 1] == -100:
                    marker = " <-- response begins"

                print(f"[{i:4d}] {token_repr:20s} | {str(label):6s} | {token_id:6d}{marker}")

            # Calculate and show token statistics after the table
            print(f"\nToken statistics:")

            # Count consecutive high-value tokens that represent the image
            # Qwen uses tokens like 151859, 151860, etc. for image patches
            image_token_threshold = 151000  # Typical threshold for Qwen image tokens
            image_token_count = np.sum(input_ids > image_token_threshold)

            # Calculate prompt tokens (everything masked)
            prompt_token_count = masked_count

            # Calculate output tokens (everything not masked)
            output_token_count = total_count - masked_count

            # Calculate non-image prompt tokens
            non_image_prompt_tokens = prompt_token_count - image_token_count

            print(f"  Image tokens: {image_token_count}")
            print(f"  Prompt tokens (total): {prompt_token_count}")
            print(f"  Prompt tokens (non-image): {non_image_prompt_tokens}")
            print(f"  Output tokens: {output_token_count}")
            print(f"  Total sequence length: {total_count}")

        # Analyze token length distribution across entire dataset
        if args.analyze_tokens and "input_ids" in sample:
            print(f"\n\n=== Analyzing token length distribution across entire dataset ===")
            print(f"Processing {len(dataset)} samples...")

            # Function to process a single sample
            def process_sample(idx):
                try:
                    current_sample = dataset[idx]
                    if "labels" in current_sample:
                        # Count total sequence length (all tokens, prompt + completion)
                        labels = current_sample["labels"]
                        total_length = len(labels)
                        return (idx, total_length, None)
                    return (idx, None, "No labels in sample")
                except Exception as e:
                    return (idx, None, str(e))

            # Process samples in parallel with progress bar
            sequence_lengths = []
            max_sequence_length = 0
            max_sequence_sample_idx = 0
            errors = []

            # Determine number of workers (use fewer workers to avoid memory issues)
            import multiprocessing

            num_workers = min(multiprocessing.cpu_count() // 2, 8)

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                futures = {executor.submit(process_sample, idx): idx for idx in range(len(dataset))}

                # Process results with progress bar
                with tqdm(total=len(dataset), desc="Analyzing samples") as pbar:
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            idx, sequence_length, error = future.result()
                            if error:
                                errors.append((idx, error))
                            elif sequence_length is not None:
                                sequence_lengths.append(sequence_length)
                                if sequence_length > max_sequence_length:
                                    max_sequence_length = sequence_length
                                    max_sequence_sample_idx = idx
                        except Exception as e:
                            errors.append((idx, f"Future error: {e}"))
                        pbar.update(1)

            if errors:
                print(f"\nEncountered {len(errors)} errors during processing")
                if len(errors) <= 5:
                    for idx, error in errors:
                        print(f"  Sample {idx}: {error}")

            if sequence_lengths:
                sequence_lengths = np.array(sequence_lengths)

                print(f"\nTotal sequence length statistics (prompt + completion):")
                print(f"  Total samples analyzed: {len(sequence_lengths)}")
                print(f"  Max sequence length: {max_sequence_length} tokens (sample index: {max_sequence_sample_idx})")
                print(f"  Min sequence length: {np.min(sequence_lengths)} tokens")
                print(f"  Mean sequence length: {np.mean(sequence_lengths):.1f} tokens")
                print(f"  Median sequence length: {np.median(sequence_lengths):.1f} tokens")
                print(f"  Std dev: {np.std(sequence_lengths):.1f} tokens")

                # Create histogram with 100-token buckets
                print(f"\nSequence length histogram (100-token buckets):")

                # Define buckets
                bucket_size = 100
                max_bucket = ((max_sequence_length // bucket_size) + 1) * bucket_size
                buckets = list(range(0, max_bucket + bucket_size, bucket_size))

                # Count samples in each bucket
                hist, _ = np.histogram(sequence_lengths, bins=buckets)

                # Find max count for scaling
                max_count = max(hist)
                bar_width = 50  # Width of histogram bars

                print(f"\n{'Range':>15} | {'Count':>6} | Distribution")
                print("-" * 80)

                for i in range(len(hist)):
                    start = buckets[i]
                    end = buckets[i + 1] - 1
                    count = hist[i]

                    # Create bar
                    if max_count > 0:
                        bar_length = int((count / max_count) * bar_width)
                        bar = "█" * bar_length
                    else:
                        bar = ""

                    range_str = f"{start:>5}-{end:>5}"
                    print(f"{range_str:>15} | {count:>6} | {bar}")

    else:
        raise AssertionError("Expected some data to be created at this point")
