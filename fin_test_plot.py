import numpy as np
import math
import csv
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

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
        self.Pf = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

    def update_step(self):
        # Update step with JPDA
        Inn = self.Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    el = math.atan(z / np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi
    az = math.atan(y / x)

    if x > 0.0:
        az = 3.14 / 2 - az
    else:
        az = 3 * 3.14 / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

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

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt))
    return measurements

def group_measurements_into_tracks(measurements):
    tracks = []
    used_indices = set()
    for i, (x_base, y_base, z_base, mt_base) in enumerate(measurements):
        if i in used_indices:
            continue
        track = [(x_base, y_base, z_base, mt_base)]
        used_indices.add(i)
        for j, (x, y, z, mt) in enumerate(measurements):
            if j in used_indices:
                continue
            if abs(mt - mt_base) < 50:
                track.append((x, y, z, mt))
                used_indices.add(j)
        tracks.append(track)
    return tracks

def is_valid_hypothesis(hypothesis):
    non_zero_hypothesis = [val for _, val in hypothesis if val != -1]
    return len(non_zero_hypothesis) == len(set(non_zero_hypothesis)) and len(non_zero_hypothesis) > 0

state_dim = 3

chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    delta = y[:3] - x[:3]
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

def perform_clustering_hypothesis_association(tracks, reports, cov_inv):
    clusters = []
    for report in reports:
        distances = [np.linalg.norm(track - report) for track in tracks]
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < chi2_threshold:
            clusters.append([min_distance_idx])

    hypotheses = []
    for cluster in clusters:
        num_tracks = len(cluster)
        base = len(reports) + 1
        for count in range(base ** num_tracks):
            hypothesis = []
            for track_idx in cluster:
                report_idx = (count // (base ** track_idx)) % base
                hypothesis.append((track_idx, report_idx - 1))
            if is_valid_hypothesis(hypothesis):
                hypotheses.append(hypothesis)

    probabilities = calculate_probabilities(hypotheses, tracks, reports, cov_inv)

    max_associations, max_probs = find_max_associations(hypotheses, probabilities, reports)

    return max_associations, max_probs

def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                distance = mahalanobis_distance(tracks[track_idx], reports[report_idx], cov_inv)
                prob *= np.exp(-0.5 * distance ** 2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return probabilities

def find_max_associations(hypotheses, probabilities, reports):
    max_associations = [-1] * len(reports)
    max_probs = [0.0] * len(reports)
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1 and prob > max_probs[report_idx]:
                max_probs[report_idx] = prob
                max_associations[report_idx] = track_idx
    return max_associations, max_probs

def main():
    kalman_filter = CVFilter()
    csv_file_path = 'ttk_84.csv'
    measurements = read_measurements_from_csv(csv_file_path)

    tracks = group_measurements_into_tracks(measurements)

    cov_inv = np.linalg.inv(np.eye(state_dim))

    for group_idx, track_group in enumerate(tracks):
        print(f"Processing group {group_idx + 1}/{len(tracks)}")

        track_states = []
        reports = []
        time_list = []
        r_list = []
        az_list = []
        el_list = []

        for x, y, z, mt in track_group:
            if len(track_states) == 0:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            else:
                kalman_filter.predict_step(mt)
                kalman_filter.initialize_measurement_for_filtering(x, y, z, mt)
                kalman_filter.update_step()

            filtered_state = kalman_filter.Sf.flatten()[:3]
            track_states.append(filtered_state)
            reports.append([x, y, z])
            time_list.append(mt)
            r_list.append(filtered_state[0])
            az_list.append(filtered_state[1])
            el_list.append(filtered_state[2])

        max_associations, max_probs = perform_clustering_hypothesis_association(track_states, reports, cov_inv)

        filtered_values_csv = np.array(track_states)

        plt.figure(figsize=(12, 6))
        plt.subplot(facecolor="white")
        plt.scatter(time_list, r_list, label='filtered range (code)', color='green', marker='*')
        plt.xlabel('Time', color='black')
        plt.ylabel('Range (r)', color='black')
        plt.title('Range vs. Time', color='black')
        plt.grid(color='gray', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.subplot(facecolor="white")
        plt.scatter(time_list, az_list, label='filtered azimuth (code)', color='green', marker='*')
        plt.xlabel('Time', color='black')
        plt.ylabel('Azimuth (az)', color='black')
        plt.title('Azimuth vs. Time', color='black')
        plt.grid(color='gray', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.subplot(facecolor="white")
        plt.scatter(time_list, el_list, label='filtered elevation (code)', color='green', marker='*')
        plt.xlabel('Time', color='black')
        plt.ylabel('Elevation (el)', color='black')
        plt.title('Elevation vs. Time', color='black')
        plt.grid(color='gray', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
