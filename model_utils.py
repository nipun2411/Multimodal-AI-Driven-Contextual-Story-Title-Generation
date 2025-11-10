sample = VIST_DATASET.shuffle().select(range(1))[0]
image, story_text, reference_title = process_vist_sample(sample)