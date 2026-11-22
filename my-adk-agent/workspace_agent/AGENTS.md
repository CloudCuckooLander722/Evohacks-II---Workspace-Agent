## 1.1 `workspace_agent`.py

### 1.11 Requirements

- The `workspace_agent` must sort incoming requests in the following categories: "DOCUMENT_CREATION", "FOLDER CREATION", "FOLDER DELETION", "DOCUMENT_MOVEMENT".

### 1.12 Test Cases

- "I want a quiz on curveball related rates problems that test my concepts" The `workspace_agent` SHOULD be able to generate and organize those problems into a Google Doc. And use similarity search to match them into a drive folder. If no exisiting drive folder, create it and place the document into the drive folder. Also ask for answer key or human confirmation before proceeding.

- "Summarize this report into a slidedeck." The `workspace_agent` SHOULD be able to analyze a document and then generate a slidedeck containing all of the important information within the document. As in the previous test case, if no exisiting drive folder, create it and place the document into the drive folder. Also ask for answer key or human confirmation before proceeding.

- "Create a PDF for me or an infographic" Same process with the test case on Google Docs.

- "Move this PDF to drive folder "Atlantic Salmon Research Project": Identifies the drive folder via similarity search, then moves the file to the drive folder. If the folder does not exist, create the folder.