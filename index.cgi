#!/usr/local/anaconda3/bin/python3

import os
import cgi
import cgitb
import cv2
import numpy as np
from urllib.parse import urlencode

cgitb.enable()

IMG_DIR = "img"
FEATURES_DIR = "features"

feature_names = ["bgr_1x1", "bgr_2x2", "bgr_3x3",
                 "hsv_1x1", "hsv_2x2", "hsv_3x3",
                 "luv_1x1", "luv_2x2", "luv_3x3",
                 "gabor", "dnn"]

distance_names = ["L2", "L1"]


def get_query_params():
    form = cgi.FieldStorage()
    feature = form.getvalue("feature", "bgr_1x1")
    distance = form.getvalue("distance", "L2")
    image_number = form.getvalue("image", "100001")
    return feature, distance, image_number


def load_features(feature_name):
    features_path = os.path.join(FEATURES_DIR, f"{feature_name}_features.npz")
    npz_data = np.load(features_path)
    return {key: npz_data[key] for key in npz_data.keys()}


def compute_l2_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))


def compute_histogram_intersection_distance(a, b):
    return np.sum(np.minimum(a, b))


def create_ranking_table(target_image_number, feature_data, distance_function):
    target_features = feature_data[target_image_number]

    image_distances = []
    for image_number, features in feature_data.items():
        if image_number == target_image_number:
            continue
        distance = distance_function(target_features, features)
        image_distances.append((image_number, distance))

    image_distances.sort(key=lambda x: x[1])
    return image_distances


def display_image(image_number, feature_name):
    image_path = os.path.join(IMG_DIR, f"{image_number}.jpg")
    query_params = urlencode({"feature": feature_name, "image": image_number})
    return f'<a href="?{query_params}"><img src="{image_path}" alt="{image_number}" style="width: 100%; max-width: 100px;"></a>'


def generate_html(feature_name, distance_name, target_image_number, ranking_table):
    html = f"Content-Type: text/html\n\n"
    html += f"""<html>
<head>
    <title>Image Search Results</title>
    <style>
        .table-container {{
            display: flex;
            flex-wrap: wrap;
        }}
        .image-table {{
            flex-basis: calc(100% / 8);
            border-collapse: collapse;
            margin: 5px;
        }}
        .image-table td {{
            border: 1px solid #ccc;
            padding: 5px;
            text-align: center;
        }}
        img {{
            width: 100%;
            max-width: 100px;
        }}
    </style>
</head>
<body>
"""

    html += "<h1>Image Search Results</h1>"

    # Add feature selection dropdown
    html += "<p>Feature:</p>"
    html += f'<form method="get" action="">'
    html += f'<input type="hidden" name="image" value="{target_image_number}">'
    html += f'<select name="feature" onchange="this.form.submit()">'
    for feature in feature_names:
        selected = "selected" if feature == feature_name else ""
        html += f'<option value="{feature}" {selected}>{feature}</option>'
    html += "</select>"

    # Add distance selection dropdown
    html += "<p>Distance:</p>"
    html += f'<select name="distance" onchange="this.form.submit()">'
    for distance in distance_names:
        selected = "selected" if distance == distance_name else ""
        html += f'<option value="{distance}" {selected}>{distance}</option>'
    html += "</select>"
    html += "</form>"

    html += f"""
    <p>Target Image:</p>
    {display_image(target_image_number, feature_name)}
    <div class="table-container">
"""

    for rank, (image_number, score) in enumerate(ranking_table, 1):
        html += f"""
        <table class="image-table">
            <tr>
                <td>Rank {rank}</td>
            </tr>
            <tr>
                <td>{display_image(image_number, feature_name)}</td>
            </tr>
            <tr>
                <td>No. {image_number}</td>
            </tr>
            <tr>
                <td>{score:.4f}</td>
            </tr>
        </table>
        """

    html += "</div></body></html>"
    return html


def main():
    feature_name, distance_name, target_image_number = get_query_params()

    feature_data = load_features(feature_name)

    distance_function = compute_l2_distance if distance_name == "L2" else compute_histogram_intersection_distance

    ranking_table = create_ranking_table(
        target_image_number, feature_data, distance_function)
    html = generate_html(feature_name, distance_name, target_image_number, ranking_table)
    print(html)


if __name__ == "__main__":
    main()

