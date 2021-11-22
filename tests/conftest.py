
_SECRET_RAW_FILE = '+sg_key = "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";\n'

_SINGLE_ADD_PATCH = (
    "diff --git a/test b/test\n"
    "new file mode 100644\n"
    "index 0000000..3c9af3f\n"
    "--- /dev/null\n"
    "+++ b/test\n"
    "@@ -0,0 +1 @@\n"
    '+sg_key = "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";\n'  # noqa
)

_SINGLE_MOVE_PATCH = (
    "diff --git a/test b/test\n"
    "index 3c9af3f..b0ce1c7 100644\n"
    "--- a/test\n"
    "+++ b/test\n"
    "@@ -1 +1,2 @@\n"
    "+something\n"
    ' sg_key = "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";\n'
)

_SINGLE_DELETE_PATCH = (
    "diff --git a/test b/test\n"
    "index b0ce1c7..deba01f 100644\n"
    "--- a/test\n"
    "+++ b/test\n"
    "@@ -1,2 +1 @@\n"
    " something\n"
    '-sg_key = "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";\n'  # noqa
)
_PATCH_WITH_NONEWLINE_BEFORE_SECRET = """
diff --git a/artifactory b/artifactory
index 2ace9c7..4c7699d 100644
--- a/artifactory
+++ b/artifactory
@@ -1,3 +1,3 @@
 some line
 some other line
-deleted line
\\ No newline at end of file
+sg_key = "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"
\\ No newline at end of file
"""


@pytest.fixture(scope="function")