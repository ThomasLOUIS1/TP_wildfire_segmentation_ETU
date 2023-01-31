import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import PIL
from PIL import Image

def display_confusion_matrix(Y_pred, Y_true):
    
    # Calculating the number of False Positives
    FP = len(np.where(Y_pred - Y_true == 1)[0])
    # Calculating the number of False Negatives
    FN = len(np.where(Y_pred - Y_true == -1)[0])
    # Calculating the number of True Positives
    TP = len(np.where(Y_pred + Y_true ==2)[0])
    # Calculating the number of True Negatives
    TN = len(np.where(Y_pred + Y_true == 0)[0])
    # Creating the confusion matrix
    cmat = [[TP, FN], [FP, TN]]
    
    # Plotting the heatmap of the confusion matrix with number of pixel
    plt.figure(figsize = (6,6))
    sns.heatmap(cmat, cmap="Reds", annot=True, fmt = 'd', square=1,   linewidth=2., xticklabels = ["Fire","No_Fire"], yticklabels = ["Fire","No_Fire"])
    plt.title("Pixel number confusion matrix")
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.show()

    # Plotting the heatmap of the confusion matrix with pixel percentage
    plt.figure(figsize = (6,6))
    sns.heatmap(cmat/np.sum(cmat), cmap="Reds", annot=True, fmt = '.02%', square=1,   linewidth=2., xticklabels = ["Fire","No_Fire"], yticklabels = ["Fire","No_Fire"])
    plt.title("Pixel percentage confusion matrix")
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.show()


def print_TPFNFPTN(Y_pred, Y_true):
    
    # Calculating the number of False Positives
    FP = len(np.where(Y_pred - Y_true == 1)[0])
    # Calculating the number of False Negatives
    FN = len(np.where(Y_pred - Y_true == -1)[0])
    # Calculating the number of True Positives
    TP = len(np.where(Y_pred + Y_true ==2)[0])
    # Calculating the number of True Negatives
    TN = len(np.where(Y_pred + Y_true == 0)[0])

    # print values
    print("TP : {} FN : {} FP : {} TN : {} ".format(TP, FN ,FP, TN))
def load_masks(paths=None, img_size=None):
    """
    Load image masks from file paths
    """
    # Initialize an array of zeros with the shape (number of masks, img_size, 1)
    y = np.zeros((len(paths),) + img_size + (1,), dtype="float32")
    
    # Loop through each mask file path
    for i, (targ_path) in enumerate(paths):
        # Load the mask image from the file path
        out = np.array(Image.open(targ_path))
        # Add an extra dimension to the mask image
        out = np.expand_dims(out, 2)
        # Set the corresponding position in the array to the mask image
        y[i] = out
    # Return the loaded masks
    return y

def print_score(score):
    """
    Print the performance score with a label
    """
    # List of score names
    score_names = ["loss", "Accuracy", "Recall", "Precision", "f1_score"]
    # Zip the score names and values
    score_and_names = zip(score_names, score)
    # Loop through each score and name pair
    for i, j in score_and_names:
        # Print the score name and value with a % format
        print("|{}: {:.02%}".format(i, j), end='|\t')

def display_matrix(matrix, title_list, path_save="image"):
    """
    Display a matrix of images with titles
    """
    # Number of columns in the matrix
    columns = len(matrix[0])
    # Number of rows in the matrix
    rows = len(matrix)
    # Create a figure with the specified number of columns and rows
    fig = plt.figure(figsize=(5 * columns, 5 * rows))
    # Loop through each row and column in the matrix
    for j in range(0, rows):
        for i in range(0, columns):
            # Add a subplot for the current image
            ax = fig.add_subplot(rows, columns, (columns * j) + i + 1)
            # Set the title for the first row
            if j == 0:
                ax.set_title(title_list[i], c='black')
            # Set the aspect ratio to be equal
            ax.set_aspect('equal')
            # Display the image
            plt.imshow(matrix[j][i])
            # Turn off the axis
            plt.axis('off')
    # Adjust the space between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # Show the plot
    plt.show()
    # Uncomment the line below to save the plot to a file
    # fig.savefig(path_save + '.jpg', bbox_inches='tight', dpi=150)

def transform_predicted_mask(pred_mask, threshhold=0.5):
    """
    Transform a predicted mask to binary format based on the threshhold value
    """
    # Set all values above the threshhold
    pred_mask[pred_mask >= threshhold] = 1.0
    pred_mask[pred_mask < threshhold] = 0.0
    return pred_mask.astype(np.uint8)


def predict(model, image, threshhold=0.5):
    """
    use the model to predict a mask from image as input
    """
    pred_mask = model.predict(image, verbose = 0)
    return transform_predicted_mask(pred_mask, threshhold=threshhold)



# A function that display firsts n_range tri-bands, masks and predictions
def display_sample_prediction(model, triband_paths, mask_paths, n_range = 3): 
  # Pass through n_range firsts images 
  i = 0
  for id in range(len(triband_paths)):
    # Read the mask data for the current iteration
    image_masks = Image.open(mask_paths[id])
    # Check if there's at least one "Fire" pixel and if the number of images displayed is less than n_range

    if np.max(image_masks) > 0 and i < n_range:
          
      # Read and normalize triband
      n_triband = np.array(Image.open(triband_paths[id]))/255.0
      
      # Add an extra dimension to the triband data for compatibility with the model input
      tribands = np.expand_dims(n_triband, 0) # Same as np.reshape(tribands, (1,256,256,3))

      # print id and path of the associated Tri-bands path
      print("Image n {}, path : {}".format(id, triband_paths[id]))
      
      predicted_mask = predict(model, tribands)
      predicted_mask = np.squeeze(predicted_mask)

      matrix=[[n_triband, image_masks, predicted_mask]] # [[Tribands,mask,predict]]

      display_matrix(matrix ,title_list=['Tri-bands', 'GT Mask', 'Predicted'])
      
      # Increment counter for number of images displayed
      i += 1
    
    # Break out of loop if n_range images have been displayed
    if i == n_range:
        break


# Used to create the TP
'''
def write_triband_and_target_from_paths(paths = None, folder = None):
    import shutil
    for i, (in_path, targ_path) in enumerate(paths):
    # copy the input image to false_color_new folder with the specified folder name
    shutil.copy(in_path, "/hdd-raid0/home_server/thomasl/Vacations/MCIA/Wildfire_segmentation/data/false_color_new/"+folder)
    # copy the target mask to masks_new folder with the specified folder name
    shutil.copy(targ_path, "/hdd-raid0/home_server/thomasl/Vacations/MCIA/Wildfire_segmentation/data/masks_new/"+folder)

# TO DO Choice the right proportion for the TP
import random

train_split = 0.3
val_split = 0.1
test_split = 0.1

combined = list(zip(triband_img_paths, target_img_paths))

random.shuffle(combined)

data_paths, label_paths = zip(*combined)

for input_path, target_path in zip(data_paths[:10], label_paths[:10]):
    print(input_path, "|", target_path)


train_index = int(len(combined) * train_split)
val_index = train_index + int(len(combined) * val_split)
test_index = val_index + int(len(combined) * test_split)

train_paths = combined[:train_index]
val_paths = combined[train_index:val_index]
test_paths = combined[val_index:test_index]

for paths in [train_data, val_data, test_data] :
  for input_path, target_path in paths[:10]:
      print(input_path, "|", target_path)
  print ("*"*30)

print("Train size : {} | Validation size : {} | Test size : {} ".format(train_index, val_index, test_index))

#############################################################################
# TEST ! DON NOT COMMIT TO GIT
#############################################################################

from tensorflow.keras import backend as K

from metrics_and_losses import recall_m, precision_m, f1_m
zeros_weight = 1.0

print("Classic bin cross")

model2 = None

model2 = Sequential()

model2.add(Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3), kernel_initializer=kernel_init, bias_initializer=bias_init))
model2.add(Conv2D(4, (3, 3), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init))

model2.add(Conv2DTranspose(4, (3, 3), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init))
model2.add(Conv2DTranspose(8, (3, 3), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init))
model2.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

metrics = [tf.keras.metrics.BinaryAccuracy(), recall_m, precision_m, f1_m]

model2.compile(optimizer = Adam(learning_rate=0.01), metrics = metrics, loss = "binary_crossentropy")

history = model2.fit(train_ds, epochs = 4, validation_data = val_ds, batch_size=batch_size, verbose=0)
model2_score = model2.evaluate(test_ds)

for ones_weight in range(1 , 250, 25):

    model2 = None

    model2 = Sequential()

    model2.add(Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3), kernel_initializer=kernel_init, bias_initializer=bias_init))
    model2.add(Conv2D(4, (3, 3), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init))

    model2.add(Conv2DTranspose(4, (3, 3), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init))
    model2.add(Conv2DTranspose(8, (3, 3), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init))
    model2.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

    #model2.summary()
    ones_weight = float(ones_weight)
    print(ones_weight)

    def weighted_binary_crossentropy( y_true, y_pred) :
        y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        logloss = -(y_true * K.log(y_pred) * ones_weight + (1 - y_true) * K.log(1 - y_pred) * zeros_weight )
        return K.mean( logloss, axis=-1)

    metrics = [tf.keras.metrics.BinaryAccuracy(), recall_m, precision_m, f1_m]

    model2.compile(optimizer = Adam(learning_rate=0.01), metrics = metrics, loss = weighted_binary_crossentropy)

    history = model2.fit(train_ds, epochs = 4, validation_data = val_ds, batch_size=batch_size, verbose=0)
    model2_score = model2.evaluate(test_ds)
    #print_score(model2_score)
#################################################################################

'''