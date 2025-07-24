from fusebuild import shell_action

shell_action(
    name="A",
    cmd="cp $OUTPUT_DIR/../C/file.txt $OUTPUT_DIR/file.txt",
    category="circular",
    tmp=None,
)

shell_action(
    name="B",
    cmd="cp $OUTPUT_DIR/../A/file.txt $OUTPUT_DIR/file.txt",
    category="circular",
    tmp=None,
)

shell_action(
    name="C",
    cmd="cp $OUTPUT_DIR/../B/file.txt $OUTPUT_DIR/file.txt",
    category="circular",
    tmp=None,
)
