from src.data import (
    download_abstract_art_dataset,
    download_coco_dataset,
    download_wikiart_dataset,
)


if __name__ == "__main__":
    download_wikiart_dataset()
    download_abstract_art_dataset()
    download_coco_dataset()
