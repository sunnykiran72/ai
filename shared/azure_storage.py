import os
import uuid
import logging
from pathlib import Path
from typing import Optional
from azure.storage.blob import BlobServiceClient, ContentSettings

logger = logging.getLogger("glamify-ai")

class AzureStorageProvider:
    """
    Unified Azure Blob Storage provider for both VTO and Wardrobe flows.
    """
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.output_container = os.getenv("AZURE_STORAGE_OUTPUT_CONTAINER", "wardrobe-outputs")
        self.input_container = os.getenv("AZURE_STORAGE_INPUT_CONTAINER", "wardrobe-inputs")
        
        if not self.connection_string:
            self.client = None
            self.account_name = None
            logger.warning("Azure Storage connection string missing. Uploads will fail.")
        else:
            self.client = BlobServiceClient.from_connection_string(self.connection_string)
            self.account_name = (
                self.connection_string.split("AccountName=")[1].split(";")[0] 
                if "AccountName=" in self.connection_string else None
            )

    def upload_image(
        self, 
        image_bytes: bytes, 
        filename: Optional[str] = None, 
        container: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> str:
        if not self.client:
            raise RuntimeError("Azure Storage client not initialized.")
        
        target_container = container or self.output_container
        
        if not filename:
            filename = f"{uuid.uuid4()}.png"
            
        if not content_type:
            suffix = Path(filename).suffix.lower().lstrip(".")
            content_type = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "webp": "image/webp",
            }.get(suffix, "image/png")

        blob_client = self.client.get_blob_client(container=target_container, blob=filename)
        upload_kwargs = {"overwrite": True}
        if content_type:
            upload_kwargs["content_settings"] = ContentSettings(content_type=content_type)
            
        blob_client.upload_blob(image_bytes, **upload_kwargs)
        
        url = f"https://{self.account_name}.blob.core.windows.net/{target_container}/{filename}"
        return url

# Singleton instance
storage = AzureStorageProvider()

if __name__ == "__main__":
    # Test script for isolated validation
    import sys
    logging.basicConfig(level=logging.INFO)
    try:
        dummy_data = b"test_image_data"
        test_url = storage.upload_image(dummy_data, filename="unified_test.txt", content_type="text/plain")
        print(f"Test Upload Successful: {test_url}")
    except Exception as e:
        print(f"Test Upload Failed: {e}")
        sys.exit(1)
