import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

# Define the measurement and track parameters
state_dim = 3  # 3D state (e.g., x, y, z)
chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    delta = y[:3] - x[:3]
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

# Covariance matrix of the measurement errors (assumed to be identity for simplicity)
cov_matrix = np.eye(state_dim)
cov_inv = np.linalg.inv(cov_matrix)

def generate_hypotheses(tracks, reports):
    num_tracks = len(tracks)
    num_reports = len(reports)
    base = num_reports + 1
    
    hypotheses = []
    for count in range(base**num_tracks):
        hypothesis = []
        for track_idx in range(num_tracks):
            report_idx = (count // (base**track_idx)) % base
            hypothesis.append((track_idx, report_idx - 1))
        
        if is_valid_hypothesis(hypothesis):
            hypotheses.append(hypothesis)
    
    return hypotheses

def is_valid_hypothesis(hypothesis):
    non_zero_hypothesis = [val for _, val in hypothesis if val != -1]
    return len(non_zero_hypothesis) == len(set(non_zero_hypothesis)) and len(non_zero_hypothesis) > 0

def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                distance = mahalanobis_distance(tracks[track_idx], reports[report_idx], cov_inv)
                prob *= np.exp(-0.5 * distance**2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return probabilities

def get_association_weights(hypotheses, probabilities):
    num_tracks = len(hypotheses[0])
    association_weights = [[] for _ in range(num_tracks)]
    
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                association_weights[track_idx].append((report_idx, prob))
    
    for track_weights in association_weights:
        track_weights.sort(key=lambda x: x[0])
        report_probs = {}
        for report_idx, prob in track_weights:
            if report_idx not in report_probs:
                report_probs[report_idx] = prob
            else:
                report_probs[report_idx] += prob
        track_weights[:] = [(report_idx, prob) for report_idx, prob in report_probs.items()]
    
    return association_weights

def find_max_associations(hypotheses, probabilities):
    max_associations = [-1] * len(reports)
    max_probs = [0.0] * len(reports)
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1 and prob > max_probs[report_idx]:
                max_probs[report_idx] = prob
                max_associations[report_idx] = track_idx
    return max_associations, max_probs

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.zeros((3, 6))  # Measurement matrix
        self.H[:, :3] = np.eye(3)
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        # print("Initialized filter state:")
        # print("Sf:", self.Sf)
        # print("Pf:", self.Pf)

    def initialize_measurement_for_filtering(self, x, y, z, mt):
        self.Z = np.array([[x], [y], [z]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        # print("Predicted filter state:")
        # print("Sf:", self.Sf)
        # print("Pf:", self.Pf)

    def update_step(self):
        # Update step with JPDA
        Inn = self.Z - np.dot(self.H, self.Sf)  # Calculate innovation directly
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        # print("Updated filter state:")
        # print("Sf:", self.Sf)
        # print("Pf:", self.Pf)

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x) * 180 / np.pi
    if az < 0:
        az += 360
    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    r = []
    el = []
    az = []
    for i in range(len(x)):
        r_i, az_i, el_i = cart2sph(x[i], y[i], z[i])
        r.append(r_i)
        az.append(az_i)
        el.append(el_i)
    return r, az, el

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt))
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'ttk_84.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

csv_file_predicted = "ttk_84.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values

A = cart2sph2(filtered_values_csv[:,1],filtered_values_csv[:,2],filtered_values_csv[:,3],filtered_values_csv)

number = 1000
result = np.divide(A[0], number)

# Lists to store the data for plotting
time_list = []
r_list = []
az_list = []
el_list = []

tracks = []
reports = []

# Iterate through measurements
for i, (x, y, z, mt) in enumerate(measurements):
    r, az, el = cart2sph(x, y, z)
    if i == 0:
        # Initialize filter state with the first measurement
        kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
    elif i == 1:
        # Initialize filter state with the second measurement and compute velocity
        prev_x, prev_y, prev_z, prev_mt = measurements[i-1]
        dt = mt - prev_mt
        vx = (x - prev_x) / dt
        vy = (y - prev_y) / dt
        vz = (z - prev_z) / dt
        kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)
    else:
        # Collect tracks and reports for association
        track_state = np.hstack((kalman_filter.Sf.flatten(), mt))
        report_state = np.hstack((x, y, z, mt))
        tracks.append(track_state)
        reports.append(report_state)

        # Chi-squared gating and clustering
        association_list = []
        for track_idx, track in enumerate(tracks):
            for report_idx, report in enumerate(reports):
                distance = mahalanobis_distance(track, report, cov_inv)
                time_diff = abs(track[3] - report[3])
                if distance < chi2_threshold and time_diff <= 50:
                    association_list.append((track_idx, report_idx))

        clusters = []
        while association_list:
            cluster_tracks = set()
            cluster_reports = set()
            stack = [association_list.pop(0)]
            while stack:
                track_idx, report_idx = stack.pop()
                cluster_tracks.add(track_idx)
                cluster_reports.add(report_idx)
                new_assoc = [(t, r) for t, r in association_list if t in cluster_tracks or r in cluster_reports]
                for assoc in new_assoc:
                    if assoc not in stack:
                        stack.append(assoc)
                association_list = [assoc for assoc in association_list if assoc not in new_assoc]

            cluster_times = set([tracks[t][3] for t in cluster_tracks] + [reports[r][3] for r in cluster_reports])
            if len(cluster_times) <= 50:
                clusters.append((list(cluster_tracks), list(cluster_reports)))

        # Process each cluster and generate hypotheses
        for track_idxs, report_idxs in clusters:
            print("Cluster Tracks:", track_idxs)
            print("Cluster Reports:", report_idxs)

            cluster_tracks = [tracks[i] for i in track_idxs]
            cluster_reports = [reports[i] for i in report_idxs]
            hypotheses = generate_hypotheses(cluster_tracks, cluster_reports)
            probabilities = calculate_probabilities(hypotheses, cluster_tracks, cluster_reports, cov_inv)
            association_weights = get_association_weights(hypotheses, probabilities)
            max_associations, max_probs = find_max_associations(hypotheses, probabilities)

            print("Hypotheses:")
            print("Tracks/Reports:", ["t" + str(i+1) for i in track_idxs])
            for hypothesis, prob  in zip(hypotheses, probabilities):
                formatted_hypothesis = ["r" + str(report_idxs[r]+1) if r != -1 else "0" for _, r in hypothesis]
                print(f"Hypothesis: {formatted_hypothesis}, Probability: {prob:.4f}")

            for track_idx, weights in enumerate(association_weights):
                for report_idx, weight in weights:
                    print(f"Track t{track_idxs[track_idx]+1}, Report r{report_idxs[report_idx]+1}: {weight:.4f}")

            for report_idx, association in enumerate(max_associations):
                if association != -1:
                    print(f"Most likely association for Report r{report_idxs[report_idx]+1}: Track t{track_idxs[association]+1}, Probability: {max_probs[report_idx]:.4f}")

            for track_idx, report_idx in zip(track_idxs, report_idxs):
                if abs(tracks[track_idx][3] - reports[report_idx][3]) <= 50:
                    print(f"Track t{track_idx+1} has Report r{report_idx+1} in the group with time less than 50 milliseconds.")
        
        # Kalman filter prediction and update steps
        kalman_filter.predict_step(mt)
        kalman_filter.initialize_measurement_for_filtering(x, y, z, mt)
        kalman_filter.update_step()

    # Append the data for plotting
    time_list.append(mt)
    r_list.append(r)
    az_list.append(az)
    el_list.append(el)

# Plot range (r) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, r_list, label='filtered range (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], result, label='filtered range (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Range (r)', color='black')
plt.title('Range vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# Plot azimuth (az) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, az_list, label='filtered azimuth (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Azimuth (az)', color='black')
plt.title('Azimuth vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# Plot elevation (el) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, el_list, label='filtered elevation (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Elevation (el)', color='black')
plt.title('Elevation vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

