from click.testing import CliRunner


@pytest.fixture(scope="session")
def cli_runner():
    os.environ["GITGUARDIAN_API_KEY"] = os.getenv(
        "TEST_GITGUARDIAN_API_KEY", "1234567890"
    )
    os.environ["GITGUARDIAN_API_URL"] = "https://api.gitguardian.com/"
    return CliRunner()


@pytest.fixture(scope="class")
def cli_fs_runner(cli_runner):
    with cli_runner.isolated_filesystem():
        yield cli_runner