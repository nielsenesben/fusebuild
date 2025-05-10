from fusebuild import *

shell_action(
    name="dependoneup",
    cmd="cat $OUTPUT_DIR/../../someaction/output1.txt > $OUTPUT_DIR/output2.txt || true",
    tmp=None,
)
