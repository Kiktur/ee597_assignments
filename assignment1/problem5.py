import numpy as np
import matplotlib.pyplot as plt
import csv

# The provided data set is taken from experiments conducted in the fourth floor of RTH. It contains
# received signal strengths from 802.11 devices located at various fixed locations. Assume that the
# signal strength decays according to the simplified path-loss model with log-normal fading. Using
# the data set, estimate the parameters for the path loss at reference distance (assume d0 =
# 1m) Kref,dB, the path loss exponent η , and the fading standard deviation 𝜎dB. The data set is
# described below and posted on the Google drive along with this assignment:

# Parameters for estimated model
d0 = 1.0 
eta = 3.3               # (3.3 really good for 10 and beyond) (2.6 good for 7-9) (3 is an ok middle ground)
kref_db = -2 
sigma_db = 1.5        # log-normal fading std dev in db (1.5 seems pretty good)
pt_db = -27          # From problem description

# Range from 1 to 50 meters
d = np.arange(1, 51)

pr_mean_dbm = (pt_db + kref_db - 10 * eta * np.log10(d / d0)) + np.random.normal(0, sigma_db, size=len(d))




transmitter_coords = []
receiver_coords = []

# Retrieve all transmitter coordinates
with open('problem5_data/transmitterXY.csv', mode='r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        transmitter_coords.append(row)

# Convert to float
float_trans_coords = [[float(item) for item in sublist] for sublist in transmitter_coords]

# Retrieve all receiver coordinates
with open('problem5_data/receiverXY.csv', mode='r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        receiver_coords.append(row)

# Convert to float
float_rec_coords = [[float(item) for item in sublist] for sublist in receiver_coords]

# Create empty array for distances between transmitters and receivers
# One row for each receiver and one column for each transmitter
rec_distances = np.zeros((len(float_rec_coords), len(float_trans_coords)))

# Calculate distances
for i in range(len(float_rec_coords)):
    for j in range(0, len(float_trans_coords)):
        dist = np.sqrt( (float_trans_coords[j][0] - float_rec_coords[i][0])**2 +  (float_trans_coords[j][1] - float_rec_coords[i][1])**2)
        rec_distances[i][j] = dist


# Need to average the received powers and reject any values that equal 500
# Convert to negative dB to plot then mess around with parameters

# Iterate through file name numbers, that's why range is weird
for k in range(7, 19):

    # Define figures starting from 1
    plt.figure(k-6)
    plt.plot(d, pr_mean_dbm, label='Estimated') # Plot estimated model first

    file_path = f'problem5_data/wifiExp{k}.csv'
    print(f"Experiment data {k}")

    # Define empty arrays for received power in experiment
    rec_data = np.zeros(len(float_rec_coords))

    # Used to keep track of valid data points (ones that aren't equal to 500)
    rec_counts = np.zeros(len(float_rec_coords))

    # If any receiver only gets invalid data points, make notes to remove it
    indices_to_remove = []

    # Get all valid data points
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            for i in range(1,len(row)): # Start from 1 to avoid timestamp data
                if (float(row[i]) != 500.0): # Reject invalid data
                    rec_data[i-1] = rec_data[i-1] + float(row[i]) # Add to total
                    rec_counts[i-1] += 1 # Keep track of valid data points

        # Get average of valid data points
        for i in range(len(rec_data)):
            if (rec_counts[i] != 0):
                rec_data[i] = rec_data[i] / rec_counts[i]
            else: # Make note to remove index if all invalid data points
                indices_to_remove.append(i)       

    # Make list of distances to plot
    plotted_rec_distances = np.zeros(len(float_rec_coords))
    for i in range(len(float_rec_coords)):
        # Get appropriate distance from receiver based on experiment data used
        plotted_rec_distances[i] = rec_distances[i][k-7]

    # For any receiver with all invalid data points, remove it from plotting
    if (len(indices_to_remove) > 0):
            rec_data = np.delete(rec_data, indices_to_remove)
            rec_counts = np.delete(rec_counts, indices_to_remove)
            plotted_rec_distances = np.delete(plotted_rec_distances, indices_to_remove) 


    plt.scatter(plotted_rec_distances, (-1 * rec_data), marker='o', label=f"Experiment data {k}")
    plt.xlabel('Distance (m)')
    plt.ylabel('Avg Received Power (dB)')
    plt.title('Received Power (dB) vs Distance')
    plt.grid(True)
    plt.legend(loc='lower left')
    

plt.show()




