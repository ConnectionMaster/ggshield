import _thread as thread
import os
import sys
import threading
from typing import Any, List

import click
from pygitguardian.models import PolicyBreak

from ggshield.dev_scan import scan_commit_range
from ggshield.filter import censor_match
from ggshield.output import OutputHandler
from ggshield.scan import ScanCollection
from ggshield.text_utils import display_error
from ggshield.utils import (
    EMPTY_SHA,
    EMPTY_TREE,
    PRERECEIVE_TIMEOUT,
    SupportedScanMode,
    handle_exception,
)

from .git_shell import get_list_commit_SHA


def quit_function() -> None:  # pragma: no cover
    display_error("\nPre-receive hook took too long")
    thread.interrupt_main()  # raises KeyboardInterrupt


class ExitAfter:
    timeout_secs: float
    timer: threading.Timer

    def __init__(self, timeout_secs: float):
        self.timeout_secs = timeout_secs

    def __enter__(self) -> None:
        if self.timeout_secs:
            self.timer = threading.Timer(self.timeout_secs, quit_function)
            self.timer.start()

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        if self.timeout_secs:
            self.timer.cancel()


class GitLabWebUIOutputHandler(OutputHandler):
    """
    Terse OutputHandler optimized for GitLab Web UI, which does not correctly handle
    multi-line texts.

    See https://docs.gitlab.com/ee/administration/server_hooks.html#custom-error-messages
    """

    def __init__(self) -> None:
        super().__init__(show_secrets=False, verbose=False)

    def echo(self, message: str, err: bool = False) -> None:
        pass

    def _process_scan_impl(self, scan_collection: ScanCollection) -> str:
        results = list(scan_collection.get_all_results())
        if not results:
            return ""

        policy_breaks = []
        for result in results:
            policy_breaks += result.scan.policy_breaks

        if len(policy_breaks) == 1:
            summary_str = "one incident"
        else:
            summary_str = f"{len(policy_breaks)} incidents"

        breaks_str = ", ".join(self.format_policy_break(x) for x in policy_breaks)
        return (
            f"GL-HOOK-ERR: ggshield found {summary_str} in these changes: {breaks_str}."
            " The commit has been rejected."
        )

    @staticmethod
    def format_policy_break(policy_break: PolicyBreak) -> str:
        """Returns a string with the policy name and a comma-separated, double-quoted,
        censored version of all `policy_break` matches.

        Looks like this:

        ("Secret Detection": "aa*******bb", "cc******dd")
        """
        matches_str = [f'"{censor_match(x)}"' for x in policy_break.matches]
        return f"({policy_break.policy}: " + ", ".join(matches_str) + ")"


def get_prereceive_timeout() -> float:
    try:
        return float(os.getenv("GITGUARDIAN_TIMEOUT", PRERECEIVE_TIMEOUT))
    except BaseException as e:
        display_error(f"Unable to parse GITGUARDIAN_TIMEOUT: {str(e)}")
        return PRERECEIVE_TIMEOUT


def get_breakglass_option() -> bool:
    """Test all options passed to git for `breakglass`"""
    raw_option_count = os.getenv("GIT_PUSH_OPTION_COUNT", None)
    if raw_option_count is not None:
        option_count = int(raw_option_count)
        for option in range(option_count):
            if os.getenv(f"GIT_PUSH_OPTION_{option}", "") == "breakglass":
                return True

    return False


@click.command()
@click.argument("prereceive_args", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--web",
    is_flag=True,
    default=None,
    help="Deprecated",
    hidden=True,
)
@click.pass_context
def prereceive_cmd(ctx: click.Context, web: bool, prereceive_args: List[str]) -> int:
    """
    scan as a pre-receive git hook.
    """
    config = ctx.obj["config"]
    out = ctx.obj["output_handler"]

    if os.getenv("GL_PROTOCOL") == "web":
        # We are inside GitLab web UI
        out = GitLabWebUIOutputHandler()

    if get_breakglass_option():
        out.echo("SKIP: breakglass detected. Skipping GitGuardian pre-receive hook.")
        return 0

    args = sys.stdin.read().strip().split()
    if len(args) < 3:
        raise click.ClickException(f"Invalid input arguments: {args}")

    before, after, *_ = args
    commit_list = []

    if after == EMPTY_SHA:
        out.echo("Deletion event or nothing to scan.")
        return 0

    if before == EMPTY_SHA:
        before = "HEAD"
        commit_list = get_list_commit_SHA(
            f"--max-count={config.max_commits_for_hook+1} {before}...{after}"
        )

        if not commit_list:
            before = EMPTY_TREE
            out.echo(
                f"New tree event. Scanning last {config.max_commits_for_hook} commits."
            )
            commit_list = get_list_commit_SHA(
                f"--max-count={config.max_commits_for_hook+1} {EMPTY_TREE} {after}"
            )
    else:
        commit_list = get_list_commit_SHA(
            f"--max-count={config.max_commits_for_hook+1} {before}...{after}"
        )

    if not commit_list:
        out.echo(
            "Unable to get commit range.\n"
            f"  before: {before}\n"
            f"  after: {after}\n"
            "Skipping pre-receive hook\n"
        )
        return 0

    if len(commit_list) > config.max_commits_for_hook:
        out.echo(
            f"Too many commits. Scanning last {config.max_commits_for_hook} commits\n"
        )
        commit_list = commit_list[-config.max_commits_for_hook :]

    if config.verbose:
        out.echo(f"Commits to scan: {len(commit_list)}")

    try:
        with ExitAfter(get_prereceive_timeout()):
            return_code = scan_commit_range(
                client=ctx.obj["client"],
                cache=ctx.obj["cache"],
                commit_list=commit_list,
                output_handler=out,
                verbose=config.verbose,
                exclusion_regexes=ctx.obj["exclusion_regexes"],
                matches_ignore=config.matches_ignore,
                all_policies=config.all_policies,
                scan_id=" ".join(commit_list),
                mode_header=SupportedScanMode.PRE_RECEIVE.value,
                banlisted_detectors=config.banlisted_detectors,
            )
            if return_code:
                out.echo(
                    """Rewrite your git history to delete evidence of your secrets.
Use environment variables to use your secrets instead and store them in a file not tracked by git.

If you don't want to go through this painful git history rewrite in the future,
you can set up ggshield in your pre commit:
https://docs.gitguardian.com/internal-repositories-monitoring/integrations/git_hooks/pre_commit

Use it carefully: if those secrets are false positives and you still want your push to pass, run:
'git push -o breakglass'""",
                    err=True,
                )
            return return_code

    except Exception as error:
        return handle_exception(error, config.verbose)
