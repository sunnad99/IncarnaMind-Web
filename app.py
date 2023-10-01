import os
from fastapi import FastAPI, UploadFile, File
from docs2db import process_files
from config import FS_PATH
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


@app.post("/upload")
async def upload_file(user_id: str, file: UploadFile = File(...)):

    # Read file contents
    file_content: bytes = file.file.read()

    # Create the path to store the file
    root_dir: str = f"{FS_PATH}{user_id}"
    pdf_data_path: str = f"{root_dir}/data"
    pdf_file_path: str = f"{pdf_data_path}/{file.filename}"
    response: dict = {"message": "File stored successfully", "status": 200}
    try:

        # Check if the pdf data directory exists
        if not os.path.exists(pdf_data_path):
            os.makedirs(pdf_data_path)

        # Write the file to the pdf data directory for the respective user
        with open(pdf_file_path, "wb") as pdf_file:
            pdf_file.write(file_content)

            print(f"File {file.filename} stored successfully")

        # Process the pdfs, create the database and store the data
        is_processed: bool = process_files(root_dir)
        if not is_processed:
            response["message"] = "File stored successfully but could not be processed"
            response["status"] = 500

        return response
    except Exception as e:
        return {"error": str(e)}
