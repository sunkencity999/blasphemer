class Blasphemer < Formula
  include Language::Python::Virtualenv

  desc "Enhanced fork of Heretic - Remove censorship from LLMs on macOS"
  homepage "https://github.com/sunkencity999/blasphemer"
  url "https://github.com/sunkencity999/blasphemer/archive/refs/tags/v1.0.1.tar.gz"
  sha256 "" # Will be filled after first release
  license "AGPL-3.0-or-later"
  head "https://github.com/sunkencity999/blasphemer.git", branch: "master"

  depends_on "cmake" => :build
  depends_on "python@3.12"

  resource "accelerate" do
    url "https://files.pythonhosted.org/packages/accelerate-1.11.0.tar.gz"
    sha256 ""
  end

  # Add other Python dependencies as resources...
  # (Homebrew will handle this automatically if we use pip install)

  def install
    # Create virtualenv
    virtualenv_install_with_resources

    # Clone and build llama.cpp as a submodule
    system "git", "submodule", "update", "--init", "--recursive"
    
    cd "llama.cpp" do
      system "cmake", "-B", "build"
      system "cmake", "--build", "build", "--config", "Release", 
             "--target", "llama-quantize", "-j", Hardware::CPU.cores
    end

    # Install helper scripts
    bin.install "blasphemer.sh"
    bin.install "convert-to-gguf.sh"
    
    # Create models directory
    (var/"blasphemer-models").mkpath
  end

  def caveats
    <<~EOS
      Blasphemer has been installed!

      Quick Start:
        1. Interactive launcher: blasphemer.sh
        2. Command line: blasphemer <model-name>
        3. Convert to GGUF: convert-to-gguf.sh <model-path>

      Documentation:
        #{opt_prefix}/USER_GUIDE.md

      Models will be saved to:
        #{var}/blasphemer-models

      Features:
        • Apple Silicon MPS GPU support
        • Automatic checkpoint/resume system
        • LM Studio GGUF conversion
        • Interactive menu-driven interface

      For help: blasphemer --help
    EOS
  end

  test do
    system "#{bin}/blasphemer", "--help"
  end
end
