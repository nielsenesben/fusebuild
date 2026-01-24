from fusebuild import *

shell_action(name="copyfile", cmd="cp file1.txt $OUTPUT_DIR/file1_copied.txt", tmp=None)

shell_action(
    name="access_non_existant_rule",
    cmd="test ! -e $OUTPUT_DIR/../nonexistant_rule/log.txt && test ! -e $OUTPUT_DIR/../nonexistant_rule",
    tmp=None,
)

shell_action(
    name="copyfile_twice",
    cmd="\n".join(
        [
            "cp file1.txt $OUTPUT_DIR/file1_copied1.txt",
            "nc localhost $TEST_PORT < /dev/zero",
            "cp file1.txt $OUTPUT_DIR/file1_copied2.txt",
        ]
    ),
    tmp=None,
    category="testspecial",
)

shell_action(name="fail", cmd="false", category="failing", tmp=None)
