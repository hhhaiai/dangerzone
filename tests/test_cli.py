from __future__ import annotations

import copy
import os
import shutil
import sys
import tempfile
import traceback
from typing import Sequence

import pytest
from click.testing import CliRunner, Result
from strip_ansi import strip_ansi  # type: ignore

from dangerzone.cli import cli_main, display_banner

from . import TestBase, for_each_doc

# TODO explore any symlink edge cases
# TODO simulate ctrl-c, ctrl-d, SIGINT/SIGKILL/SIGTERM... (man 7 signal), etc?
# TODO validate output PDFs https://github.com/pdfminer/pdfminer.six
# TODO trigger "Invalid json returned from container"
# TODO trigger "pdf-to-pixels failed"
# TODO simulate container runtime missing
# TODO simulate container connection error
# TODO simulate container connection loss
# FIXME "/" path separator is platform-dependent, use pathlib instead


class CLIResult(Result):
    """Wrapper class for Click results.

    This class wraps Click results and provides the following extras:
    * Assertion statements for success/failure.
    * Printing the result details, when an assertion fails.
    * The arguments of the invocation, which are not provided by the stock
    `Result` class.
    """

    @classmethod
    def reclass_click_result(cls, result: Result, args: Sequence[str]) -> CLIResult:
        result.__class__ = cls
        result.args = copy.deepcopy(args)  # type: ignore[attr-defined]
        return result  # type: ignore[return-value]

    def assert_success(self) -> None:
        """Assert that the command succeeded."""
        try:
            assert self.exit_code == 0
            assert self.exception is None
        except AssertionError:
            self.print_info()
            raise

    def assert_failure(self, exit_code: int = None, message: str = None) -> None:
        """Assert that the command failed.

        By default, check that the command has returned with an exit code
        other than 0. Alternatively, the caller can check for a specific exit
        code. Also, the caller can check if the output contains an error
        message.
        """
        try:
            if exit_code is None:
                assert self.exit_code != 0
            else:
                assert self.exit_code == exit_code
            if message is not None:
                assert message in self.output
        except AssertionError:
            self.print_info()
            raise

    def print_info(self) -> None:
        """Print all the info we have for a CLI result.

        Print the string representation of the result, as well as:
        1. Command output (if any).
        2. Exception traceback (if any).
        """
        print(self)
        num_lines = len(self.output.splitlines())
        if num_lines > 0:
            print(f"Output ({num_lines} lines follow):")
            print(self.output)
        else:
            print("Output (0 lines).")

        if self.exc_info:
            print("The original traceback follows:")
            traceback.print_exception(*self.exc_info, file=sys.stdout)

    def __str__(self) -> str:
        """Return a string representation of a CLI result.

        Include the arguments of the command invocation, as well as the exit
        code and exception message.
        """
        desc = (
            f"<CLIResult args: {self.args},"  # type: ignore[attr-defined]
            f" exit code: {self.exit_code}"
        )
        if self.exception:
            desc += f", exception: {self.exception}"
        return desc


class TestCli(TestBase):
    def run_cli(self, args: Sequence[str] | str = ()) -> CLIResult:
        """Run the CLI with the provided arguments.

        Callers can either provide a list of arguments (iterable), or a single
        argument (str). Note that in both cases, we don't tokenize the input
        (i.e., perform `shlex.split()`), as this is prone to errors in Windows
        environments [1]. The user must perform the tokenizaton themselves.

        [1]: https://stackoverflow.com/a/35900070c
        """
        if isinstance(args, str):
            # Convert the single argument to a tuple, else Click will attempt
            # to tokenize it.
            args = (args,)
        result = CliRunner().invoke(cli_main, args)
        return CLIResult.reclass_click_result(result, args)


class TestCliBasic(TestCli):
    def test_no_args(self):
        """``$ dangerzone-cli``"""
        result = self.run_cli()
        result.assert_failure()

    def test_help(self):
        """``$ dangerzone-cli --help``"""
        result = self.run_cli("--help")
        result.assert_success()

    def test_display_banner(self, capfd):
        display_banner()  # call the test subject
        (out, err) = capfd.readouterr()
        plain_lines = [strip_ansi(line) for line in out.splitlines()]
        assert "╭──────────────────────────╮" in plain_lines, "missing top border"
        assert "╰──────────────────────────╯" in plain_lines, "missing bottom border"

        banner_width = len(plain_lines[0])
        for line in plain_lines:
            assert len(line) == banner_width, "banner has inconsistent width"


class TestCliConversion(TestCliBasic):
    def test_invalid_lang(self):
        result = self.run_cli([self.sample_doc, "--ocr-lang", "piglatin"])
        result.assert_failure()

    @for_each_doc
    def test_formats(self, doc):
        result = self.run_cli(str(doc))
        result.assert_success()

    def test_output_filename(self):
        temp_dir = tempfile.mkdtemp(prefix="dangerzone-")
        result = self.run_cli(
            [self.sample_doc, "--output-filename", f"{temp_dir}/safe.pdf"]
        )
        result.assert_success()

    def test_output_filename_spaces(self):
        temp_dir = tempfile.mkdtemp(prefix="dangerzone-")
        result = self.run_cli(
            [self.sample_doc, "--output-filename", f"{temp_dir}/safe space.pdf"]
        )
        result.assert_success()

    def test_output_filename_new_dir(self):
        result = self.run_cli(
            [self.sample_doc, "--output-filename", "fake-directory/my-output.pdf"]
        )
        result.assert_failure()

    def test_sample_not_found(self):
        result = self.run_cli("fake-directory/fake-file.pdf")
        result.assert_failure()

    def test_lang_eng(self):
        result = self.run_cli([self.sample_doc, "--ocr-lang", "eng"])
        result.assert_success()

    @pytest.mark.parametrize(
        "filename,",
        [
            "“Curly_Quotes”.pdf",  # issue 144
            "Оригинал.pdf",
            "spaces test.pdf",
        ],
    )
    def test_filenames(self, filename):
        tempdir = tempfile.mkdtemp(prefix="dangerzone-")
        doc_path = os.path.join(filename)
        shutil.copyfile(self.sample_doc, doc_path)
        result = self.run_cli(doc_path)
        shutil.rmtree(tempdir)
        result.assert_success()