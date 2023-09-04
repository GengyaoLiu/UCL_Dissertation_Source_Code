import os
import pandas as pd
from PIL import Image
from torchvision import transforms




def augment(df, transform, dataset):
    # Directory with original images
    original_dir = '/content/gdrive/MyDrive/Dissertation/dataset/Dog Emotion/images'

    # Directory to save augmented images
    augmented_dir = '/content/gdrive/MyDrive/Dissertation/dataset/Dog Emotion/images_augment'

    # Check dataset
    if dataset not in ["train", "validate"]:
      assert AssertionError("Parameter dataset must be either train or validate")

    new_label_dir = os.path.join('/content/gdrive/MyDrive/Dissertation/dataset/Dog Emotion', dataset+'_labels.csv')

    # Prepare dataframe for new labels
    new_labels_df = pd.DataFrame(columns=['filename', 'label'])

    # Augment images
    if not os.path.exists(new_label_dir):
      if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)
      for idx, row in df.iterrows():
          print("\n Loading a new image for augmentation... \n")

          # Get the file name and original label for this image
          filename = row[1]
          original_label = row[2]
          # Open an image file
          image = Image.open(os.path.join(original_dir, filename))
          # Apply the transformations and save the transformed image
          print("\n Creating nine augmented versions...\n")
          for i in range(9):  # Creating nine augmented versions
              transformed_image = transform(image)
              new_filename = f"{filename.split('.')[0]}_augmented{i}.{filename.split('.')[1]}"
              transformed_image.save(os.path.join(augmented_dir, new_filename))
              # Append to new labels dataframe
              new_labels_df = new_labels_df.append({'filename': new_filename, 'label': original_label}, ignore_index=True)

          print("\n Finished creating nine augmented versions!!!!!!!\n")

          # Save the original image and label
          new_labels_df = new_labels_df.append({'filename': filename, 'label': original_label}, ignore_index=True)
          image.save(os.path.join(augmented_dir, filename))
          print("Original image was saved!!!!")
      # Save new labels to a CSV file
      new_labels_df.to_csv(os.path.join(new_label_dir), index=False)
    else:
      print("\n There is already an augmented dataset, keep using it! \n")
      # Load new labels to a CSV file
      new_labels_df = pd.read_csv(new_label_dir)
    return new_labels_df




