# Learnt Lessons

- For this u6gb environment, the Notion connector token path should be verified live before reuse because workspace aliases may not contain the token file.
- The correct path evidence is file metadata only; the secret value itself should not be read or exposed.
- For Notion answer pages in this workspace, use real `<callout>` blocks and re-fetch the page after writeback.
