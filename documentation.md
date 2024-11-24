### 1. **Explanation of Each Function**

1. **`convert_to_base64(image)`**
   - Converts an image to a base64-encoded string. Useful for embedding images in JSON responses or displaying them in web pages.

2. **`DocumentProcessor.__init__(self, azure_key, azure_endpoint)`**
   - Initializes the `DocumentProcessor` class with Azure OCR client, a SpaCy NER model, and Tesseract OCR configuration.

3. **`DocumentProcessor.process_document(self, file_path)`**
   - Main entry point to process a document. Determines if the input is a PDF or an image and calls the appropriate processing method.

4. **`DocumentProcessor.process_pdf(self, pdf_path)`**
   - Processes multi-page PDF documents. Converts pages to images, processes them individually, and aggregates the results.

5. **`DocumentProcessor.process_image(self, image)`**
   - Processes a single image. Handles preprocessing, text extraction, structured information extraction, table detection, and signature detection.

6. **`DocumentProcessor.preprocess_image(self, image)`**
   - Enhances the image by converting to grayscale, applying thresholding, denoising, and correcting skew.

7. **`DocumentProcessor.extract_text(self, image)`**
   - Extracts text from an image using Tesseract OCR and Azure OCR, then merges and cleans the results.

8. **`DocumentProcessor.extract_structured_info(self, text)`**
   - Uses NLP techniques to extract entities (dates, emails, phone numbers, addresses) from raw text.

9. **`DocumentProcessor.detect_tables(self, image)`**
   - Detects table-like structures in the image, extracts their content, and formats it into structured data.

10. **`DocumentProcessor.detect_signatures(self, image)`**
    - Identifies signature-like regions in the image using morphological operations and contour analysis.

11. **`DocumentProcessor.extract_text_azure(self, image)`**
    - Uses Azure OCR to extract text from an image. Handles API calls and potential errors.

12. **`DocumentProcessor.merge_text_results(self, text_list)`**
    - Merges text results from multiple OCR sources, removes duplicates, and cleans artifacts.

13. **`DocumentProcessor.extract_dates(self, text)`**
    - Identifies and parses dates from text using regex and `dateparser`.

14. **`DocumentProcessor.extract_emails(self, text)`**
    - Extracts email addresses using regex.

15. **`DocumentProcessor.extract_phone_numbers(self, text)`**
    - Extracts phone numbers using regex patterns.

16. **`DocumentProcessor.extract_addresses(self, text)`**
    - Extracts addresses using regex and NLP entity recognition.

17. **`DocumentProcessor.get_skew_angle(self, image)`**
    - Computes the skew angle of the document for deskewing.

18. **`DocumentProcessor.rotate_image(self, image, angle)`**
    - Rotates an image by a specified angle.

19. **`DocumentProcessor.group_lines_into_tables(self, lines)`**
    - Groups detected lines into potential table boundaries.

20. **`DocumentProcessor.check_table_bounds(self, h1, h2, v1, v2)`**
    - Checks if four lines form a valid table.

21. **`DocumentProcessor.lines_intersect(self, line1, line2)`**
    - Determines if two lines intersect.

22. **`DocumentProcessor.merge_overlapping_tables(self, tables)`**
    - Merges overlapping table boundaries.

23. **`DocumentProcessor.structure_table_data(self, table_df)`**
    - Converts raw table data into a structured format.

24. **`DocumentProcessor.is_signature_like(self, contour)`**
    - Checks if a contour resembles a signature.

25. **`DocumentProcessor.calculate_signature_confidence(self, signature_image)`**
    - Assigns a confidence score for detected signatures based on contour properties.

26. **`process_document_batch(input_path, output_path, azure_key, azure_endpoint)`**
    - Processes a batch of documents from a directory and saves the results.

---

### 2. **Potential Sources of Errors**

- **`convert_to_base64`**: Image encoding might fail for corrupted images.
- **Azure OCR Integration**:
  - API key or endpoint misconfiguration.
  - Network issues or Azure service downtime.
  - Response format changes or errors in Azure's API.
- **`process_document`**:
  - Unsupported file types.
  - Failure to read or load the file.
- **Image Preprocessing (`preprocess_image`)**:
  - Denoising or skew angle computation might fail for low-quality or highly skewed images.
- **Table Detection (`detect_tables`)**:
  - Incorrect grouping of lines.
  - Misidentification of non-table structures as tables.
- **Text Extraction (`extract_text`)**:
  - Inconsistent results from Tesseract or Azure OCR for low-quality images.
  - Tesseract misconfigurations.
- **Regex-Based Extraction**:
  - Missed patterns or false positives due to poorly formatted text.
- **NLP-Based Extraction**:
  - Incorrect entity recognition or missing SpaCy model.
- **Batch Processing (`process_document_batch`)**:
  - Directory not found or inaccessible files.
  - JSON writing errors if results contain non-serializable data.
- **Signature Detection (`detect_signatures`)**:
  - Contours falsely identified as signatures.
  - Errors in calculating confidence scores.

---

### 3. **Flow/Structure of the Script**

1. **Initialization**:
   - Load dependencies.
   - Set up the Azure OCR client and SpaCy NLP model.

2. **Main Entry Points**:
   - `process_document` determines the input type (PDF or image).
   - Redirects to specific handlers (`process_pdf` or `process_image`).

3. **Image Processing**:
   - Preprocess the image for better text and structure detection.

4. **Text Extraction**:
   - Use Tesseract and Azure OCR to extract text.
   - Merge and clean results.

5. **Information Extraction**:
   - Use regex and NLP techniques to extract structured data (entities, dates, etc.).

6. **Content Analysis**:
   - Detect tables and extract structured data.
   - Identify and assess signatures.

7. **Batch Processing**:
   - Handle multiple files in a directory.
   - Save results in a structured JSON file.

8. **Error Handling**:
   - Handle API errors, unsupported file types, and other exceptions gracefully.

9. **Helper Methods**:
   - Support functions for preprocessing, table merging, signature detection, and more.
