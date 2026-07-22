import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from heretic.utils import print


class Deployer:
    # Candidate locations for the llama-quantize binary, relative to the
    # llama.cpp root. Modern llama.cpp (CMake) produces build/bin/llama-quantize;
    # older layouts are kept as fallbacks.
    QUANTIZE_CANDIDATES = (
        "build/bin/llama-quantize",
        "build/bin/quantize",
        "llama-quantize",
        "quantize",
    )

    def __init__(self, llama_cpp_root: str = "llama.cpp"):
        self.llama_cpp_root = Path(llama_cpp_root).resolve()

    def _find_quantize_binary(self) -> Optional[Path]:
        """Locate an existing llama-quantize binary, if any."""
        for candidate in self.QUANTIZE_CANDIDATES:
            path = self.llama_cpp_root / candidate
            if path.exists():
                return path
        return None

    def ensure_llama_cpp_build(self) -> bool:
        """
        Ensure llama.cpp tools are built and available.
        Returns True if successful, False otherwise.
        """
        if not self.llama_cpp_root.exists():
            print(f"[red]Error: llama.cpp directory not found at {self.llama_cpp_root}[/]")
            print("[yellow]Did you clone with --recursive? Try: git submodule update --init --recursive[/]")
            return False

        if self._find_quantize_binary() is not None:
            return True

        # Modern llama.cpp uses CMake; the old `make quantize` target no longer exists.
        cmake = shutil.which("cmake")
        if cmake is None:
            print("[red]Error: 'cmake' not found on PATH. Install it (e.g. 'brew install cmake') to build llama.cpp.[/]")
            return False

        print("[yellow]llama.cpp quantize tool not found. Building with CMake (this may take a few minutes)...[/]")
        build_dir = self.llama_cpp_root / "build"
        try:
            subprocess.run(
                [cmake, "-B", str(build_dir), "-S", str(self.llama_cpp_root)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    cmake,
                    "--build",
                    str(build_dir),
                    "--config",
                    "Release",
                    "--target",
                    "llama-quantize",
                    "-j",
                    "8",
                ],
                check=True,
                capture_output=True,
            )
            print("[green]Successfully built llama.cpp quantize tool[/]")
        except subprocess.CalledProcessError as e:
            print(f"[red]Error building llama.cpp: {e}[/]")
            if e.stderr:
                print(f"[red]{e.stderr.decode(errors='replace')}[/]")
            return False

        if self._find_quantize_binary() is None:
            print("[red]Build finished but llama-quantize binary was not found.[/]")
            return False

        return True

    def convert_to_gguf(
        self,
        model_path: Path,
        output_path: Optional[Path] = None,
        out_type: str = "f16",
    ) -> Optional[Path]:
        """
        Convert a HF model to GGUF format.

        Args:
            model_path: Path to the model directory
            output_path: Optional path for the output GGUF file
            out_type: Output type (f16, f32, etc.)

        Returns:
            Path to the generated GGUF file, or None if failed.
        """
        convert_script = self.llama_cpp_root / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            print(f"[red]Error: Conversion script not found at {convert_script}[/]")
            return None

        if output_path is None:
            output_path = model_path / f"{model_path.name}-{out_type}.gguf"

        print(f"[cyan]Converting model to GGUF ({out_type})...[/]")

        cmd = [
            sys.executable,
            str(convert_script),
            str(model_path),
            "--outfile",
            str(output_path),
            "--outtype",
            out_type,
        ]

        try:
            # Assumes the current environment has the necessary deps
            # (torch, transformers, etc.), which Blasphemer requires anyway.
            subprocess.run(cmd, check=True)
            print(f"[green]Conversion successful: {output_path}[/]")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"[red]Conversion failed: {e}[/]")
            return None

    def quantize_model(
        self,
        gguf_path: Path,
        methods: List[str],
    ) -> List[Path]:
        """
        Quantize a GGUF file using specified methods.

        Args:
            gguf_path: Path to the source GGUF file (usually f16)
            methods: List of quantization methods (e.g. ["Q4_K_M", "Q5_K_M"])

        Returns:
            List of paths to quantized files.
        """
        if not self.ensure_llama_cpp_build():
            return []

        quantize_bin = self._find_quantize_binary()
        if quantize_bin is None:
            print("[red]Error: llama-quantize binary not available.[/]")
            return []

        output_files = []

        for method in methods:
            output_file = gguf_path.parent / f"{gguf_path.stem.replace('-f16', '')}-{method}.gguf"
            print(f"[cyan]Quantizing to {method}...[/]")

            cmd = [
                str(quantize_bin),
                str(gguf_path),
                str(output_file),
                method,
            ]

            try:
                subprocess.run(cmd, check=True)
                print(f"[green]Quantization successful: {output_file}[/]")
                output_files.append(output_file)
            except subprocess.CalledProcessError as e:
                print(f"[red]Quantization failed for {method}: {e}[/]")

        return output_files
