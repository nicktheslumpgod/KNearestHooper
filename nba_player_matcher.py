import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Optional, Union, Tuple, Any
import os
import re

#helper function to to convert numpt types to native python for JSON serialization
def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.generic)):
        return obj.item()
    else:
        return obj

class EnhancedNBAPlayerMatcher:
    """
    An improved matcher to find NBA players similar to user profiles based on shot types, 
    court zones, and height with PCA dimensionality reduction and weighted feature importance.
    """
    
    def __init__(self, data_path: str, percentile_path: Optional[str] = None, 
                 pca_variance_threshold: float = 0.95, use_pca: bool = True):
        """
        Initialize the Enhanced NBA Player Matcher with PCA and weighted KNN features.
        
        Parameters:
        data_path (str): Path to the CSV file with processed NBA player data
        percentile_path (str, optional): Path to a JSON file with percentiles for scaling
        pca_variance_threshold (float): Threshold for explained variance in PCA (default 0.95)
        use_pca (bool): Whether to use PCA for dimensionality reduction (default True)
        """
        # Load the processed NBA player data
        self.player_data = pd.read_csv(data_path)
        
        # Store PCA configuration
        self.use_pca = use_pca
        self.pca_variance_threshold = pca_variance_threshold
        
        # Load percentiles if available (for scaling API inputs)
        self.percentiles = None
        if percentile_path and os.path.exists(percentile_path):
            try:
                with open(percentile_path, 'r') as f:
                    self.percentiles = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load percentiles from {percentile_path}: {e}")
        
        # Define feature categories for balanced weighting
        self.shot_type_cols = ["Dunk", "Driving", "Jumpshot", "Floater", "Step Back", "Hook", "Fadeaway"]
        
        self.zone_cols = [
            "Restricted Area - Center", "In The Paint (Non-RA) - Center", "Mid-Range - Center",
            "Mid-Range - Left Side", "Mid-Range - Right Side", "Mid-Range - Left Side Center", 
            "Mid-Range - Right Side Center", "Left Corner 3 - Left Side", "Right Corner 3 - Right Side",
            "Above the Break 3 - Left Side Center", "Above the Break 3 - Right Side Center", 
            "Above the Break 3 - Center", "In The Paint (Non-RA) - Left Side", "In The Paint (Non-RA) - Right Side"
        ]
        
        # All feature columns for KNN (shot types and zones)
        self.feature_cols = self.shot_type_cols + self.zone_cols
        
        # Create scalers for each feature category
        self.shot_type_scaler = MinMaxScaler()
        self.zone_scaler = MinMaxScaler()
        
        # Initialize PCA components
        self.shot_type_pca = None
        self.zone_pca = None
        self.n_shot_components = None
        self.n_zone_components = None
        
        # Verify that the player data contains our expected columns
        missing_shot_cols = [col for col in self.shot_type_cols if col not in self.player_data.columns]
        missing_zone_cols = [col for col in self.zone_cols if col not in self.player_data.columns]
        
        if missing_shot_cols or missing_zone_cols:
            print(f"Warning: Missing expected columns in player data:")
            if missing_shot_cols:
                print(f"Missing shot type columns: {missing_shot_cols}")
            if missing_zone_cols:
                print(f"Missing zone columns: {missing_zone_cols}")
                
        # Fill any missing columns with zeros
        for col in self.shot_type_cols + self.zone_cols:
            if col not in self.player_data.columns:
                self.player_data[col] = 0
        
        # Process and prepare the data
        self._prepare_data()
        
        # Define category weights for balanced contribution
        self.category_weights = {
            "height": 0.33,  # Height gets 1/3 of the weight
            "shot_types": 0.33,  # Shot types get 1/3 of the weight 
            "zones": 0.33  # Court zones get 1/3 of the weight
        }
        
        # Initialize heights for all NBA players
        self.player_heights = self._process_player_heights()
        
        # Initialize base KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=10,  # Get more neighbors than needed to filter later
            algorithm="brute",  # Use brute force for custom metric
            metric="euclidean"  # Base metric - we'll apply weights manually
        )
        
        # Fit the model to our feature matrix
        self.knn_model.fit(self.feature_matrix)
        
        # Define mappings from API shot types to feature column names
        self.shot_mapping = {
            "shot-dunk": "Dunk",
            "shot-layup": "Driving",  # Map layup to Driving as best match
            "shot-jumpshot": "Jumpshot",
            "shot-floater": "Floater",
            "shot-stepback": "Step Back",
            "shot-hook": "Hook",
            "shot-fadeaway": "Fadeaway"
        }
        
        # Define mappings from API court zone IDs to feature column names
        self.zone_mapping = {
            "restricted-area": "Restricted Area - Center",
            "low-paint": "In The Paint (Non-RA) - Center",
            "high-paint": "Mid-Range - Center",
            "left-corner-mid": "Mid-Range - Left Side",
            "right-corner-mid": "Mid-Range - Right Side",
            "left-wing-mid": "Mid-Range - Left Side Center",
            "right-wing-mid": "Mid-Range - Right Side Center",
            "top-key-mid": "Mid-Range - Center",
            "left-corner-three": "Left Corner 3 - Left Side",
            "right-corner-three": "Right Corner 3 - Right Side",
            "left-wing-three": "Above the Break 3 - Left Side Center",
            "right-wing-three": "Above the Break 3 - Right Side Center",
            "top-key-three": "Above the Break 3 - Center"
        }
        
        # Map API intensity values to percentile labels
        self.intensity_mapping = {
            "none": "Low",
            "low": "Below Average",
            "medium": "Average",
            "high": "High"
        }
        
        # List of per-game stat columns to include in the output
        self.per_game_stats = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 
                              'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 
                              'STL', 'BLK', 'TOV', 'PF', 'PTS']
    
    def _prepare_data(self):
        """
        Prepare the data by scaling features and applying PCA if enabled.
        This method is called during initialization.
        """
        # Make sure we have all columns in the right order
        shot_type_df = self.player_data[self.shot_type_cols].copy()
        zone_df = self.player_data[self.zone_cols].copy()
        
        # Fit and transform the data with scalers
        shot_type_features_scaled = self.shot_type_scaler.fit_transform(shot_type_df)
        zone_features_scaled = self.zone_scaler.fit_transform(zone_df)
        
        # Store the original scaled features
        self.shot_type_features_scaled = shot_type_features_scaled
        self.zone_features_scaled = zone_features_scaled
        
        if self.use_pca:
            # Apply PCA to shot types
            self.shot_type_pca = PCA(n_components=min(len(self.shot_type_cols), len(self.player_data)-1))
            shot_type_pca_features = self.shot_type_pca.fit_transform(shot_type_features_scaled)
            
            # Apply PCA to zones
            self.zone_pca = PCA(n_components=min(len(self.zone_cols), len(self.player_data)-1))
            zone_pca_features = self.zone_pca.fit_transform(zone_features_scaled)
            
            # Determine optimal number of components based on explained variance
            self.n_shot_components = self._get_optimal_components(self.shot_type_pca.explained_variance_ratio_)
            self.n_zone_components = self._get_optimal_components(self.zone_pca.explained_variance_ratio_)
            
            # Print PCA information
            print(f"Shot types: Using {self.n_shot_components} components out of {shot_type_features_scaled.shape[1]} " +
                  f"({self.shot_type_pca.explained_variance_ratio_[:self.n_shot_components].sum()*100:.2f}% variance)")
            print(f"Zones: Using {self.n_zone_components} components out of {zone_features_scaled.shape[1]} " +
                  f"({self.zone_pca.explained_variance_ratio_[:self.n_zone_components].sum()*100:.2f}% variance)")
            
            # Store PCA-transformed features (truncated to optimal components)
            self.shot_type_pca_features = shot_type_pca_features[:, :self.n_shot_components]
            self.zone_pca_features = zone_pca_features[:, :self.n_zone_components]
            
            # Store feature loading matrices for reverse mapping
            self.shot_type_loadings = self.shot_type_pca.components_[:self.n_shot_components]
            self.zone_loadings = self.zone_pca.components_[:self.n_zone_components]
            
            # Combine all PCA features for the KNN model
            self.feature_matrix = np.hstack([self.shot_type_pca_features, self.zone_pca_features])
        else:
            # Use scaled features directly if PCA is disabled
            self.feature_matrix = np.hstack([shot_type_features_scaled, zone_features_scaled])
    
    def _get_optimal_components(self, explained_variance_ratio):
        """
        Determine the optimal number of principal components based on explained variance.
        
        Parameters:
        explained_variance_ratio (np.array): Array of explained variance ratios
        
        Returns:
        int: Optimal number of components
        """
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components = np.argmax(cumulative_variance >= self.pca_variance_threshold) + 1
        return max(n_components, 1)  # Ensure at least one component
    
    def visualize_explained_variance(self):
        """
        Visualize the explained variance ratio for shot types and zones.
        Creates a plot showing how much variance is explained by each principal component.
        """
        if not self.use_pca or self.shot_type_pca is None or self.zone_pca is None:
            print("PCA not initialized. Cannot visualize explained variance.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Shot types
        n_components = len(self.shot_type_pca.explained_variance_ratio_)
        x = range(1, n_components + 1)
        
        ax1.bar(x, self.shot_type_pca.explained_variance_ratio_, alpha=0.5, color='b')
        ax1.plot(x, np.cumsum(self.shot_type_pca.explained_variance_ratio_), 'r-')
        ax1.axhline(y=self.pca_variance_threshold, color='g', linestyle='--')
        ax1.axvline(x=self.n_shot_components, color='r', linestyle='--')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Shot Types: Explained Variance by Component')
        ax1.grid(True)
        
        # Zones
        n_components = len(self.zone_pca.explained_variance_ratio_)
        x = range(1, n_components + 1)
        
        ax2.bar(x, self.zone_pca.explained_variance_ratio_, alpha=0.5, color='b')
        ax2.plot(x, np.cumsum(self.zone_pca.explained_variance_ratio_), 'r-')
        ax2.axhline(y=self.pca_variance_threshold, color='g', linestyle='--')
        ax2.axvline(x=self.n_zone_components, color='r', linestyle='--')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Explained Variance Ratio')
        ax2.set_title('Zones: Explained Variance by Component')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _process_player_heights(self) -> Dict[str, float]:
        """
        Process player heights from the dataset.
        
        Returns:
        Dict[str, float]: Dictionary mapping player names to normalized height values
        """
        # In a real implementation, we would have actual player heights
        # For now, create synthetic height data based on position or use a placeholder
        player_heights = {}
        
        # Assuming we might have a 'HEIGHT' or 'height' column, or we could estimate from position
        if 'HEIGHT' in self.player_data.columns:
            for _, player in self.player_data.iterrows():
                player_heights[player['PLAYER_NAME']] = float(player['HEIGHT'])
        elif 'height' in self.player_data.columns:
            for _, player in self.player_data.iterrows():
                player_heights[player['PLAYER_NAME']] = float(player['height'])
        else:
            # Estimate based on position if available
            position_heights = {
                'PG': 0.2,  # Point guards (shortest)
                'SG': 0.4,  # Shooting guards
                'SF': 0.6,  # Small forwards
                'PF': 0.8,  # Power forwards
                'C': 1.0    # Centers (tallest)
            }
            
            if 'POSITION' in self.player_data.columns:
                for _, player in self.player_data.iterrows():
                    pos = player['POSITION'] if pd.notna(player['POSITION']) else 'SF'  # Default
                    player_heights[player['PLAYER_NAME']] = position_heights.get(pos, 0.5)
            else:
                # Fallback: assign average height to all players
                for _, player in self.player_data.iterrows():
                    player_heights[player['PLAYER_NAME']] = 0.5
        
        return player_heights
    
    def _map_feature_weights_to_pca(self, feature_weights: Dict[str, float], feature_type: str) -> np.ndarray:
        """
        Map feature weights from original space to PCA space based on component loadings.
        
        Parameters:
        feature_weights (Dict[str, float]): Weights for original features
        feature_type (str): Type of features ('shot_types' or 'zones')
        
        Returns:
        np.ndarray: Weights for PCA components
        """
        if not self.use_pca:
            return feature_weights
        
        if feature_type == 'shot_types':
            # Get the original feature weights in the correct order
            weights_array = np.array([feature_weights.get(col, 0.1) for col in self.shot_type_cols])
            # Use the absolute loadings and the original weights to compute component weights
            # We use abs() because the sign in PCA loadings represents direction, not importance
            component_weights = np.abs(self.shot_type_loadings) @ weights_array
            # Normalize to keep the overall weighting scale
            if np.sum(component_weights) > 0:
                component_weights = component_weights / np.sum(component_weights) * np.sum(weights_array)
            return component_weights
        
        elif feature_type == 'zones':
            # Get the original feature weights in the correct order
            weights_array = np.array([feature_weights.get(col, 0.1) for col in self.zone_cols])
            # Use the absolute loadings and the original weights to compute component weights
            component_weights = np.abs(self.zone_loadings) @ weights_array
            # Normalize to keep the overall weighting scale
            if np.sum(component_weights) > 0:
                component_weights = component_weights / np.sum(component_weights) * np.sum(weights_array)
            return component_weights
        
        return np.array([])
    
    def _get_player_image_url(self, player_name: str) -> str:
        """
        Generate the image URL for a player based on their name.
        
        Parameters:
        player_name (str): The name of the player
        
        Returns:
        str: The URL to the player's image
        """
        # Create a sanitized filename from the player name
        # Remove special characters and replace spaces with underscores
        sanitized_name = re.sub(r'[^\w\s]', '', player_name).lower().replace(' ', '_')
        
        # Get the base URL from environment variable or use default
        base_url = os.environ.get("BASE_URL", "")
        
        # Construct the image path
        image_path = f"{base_url}/images/{sanitized_name}.jpg"
        
        return image_path
    
    def preprocess_api_input(self, api_data: Dict) -> Tuple[pd.Series, Dict]:
        """
        Transform API input data to match the structure of the KNN model with feature weighting.
        
        Parameters:
        api_data (dict): The API request payload with player, shots, and zones
        
        Returns:
        Tuple[pd.Series, Dict]: A feature vector and a dictionary of feature weights
        """
        # Create empty feature vectors for each category
        shot_features = pd.Series(0.0, index=self.shot_type_cols)
        zone_features = pd.Series(0.0, index=self.zone_cols)
        
        # Track user preference weights for each feature
        feature_weights = {}
        
        # Process height (convert to normalized value)
        height_value = 0.5  # Default to medium height
        if "player" in api_data and "height" in api_data["player"]:
            height = api_data["player"]["height"].lower()
            if height == "short":
                height_value = 0.2
            elif height == "average":
                height_value = 0.5
            elif height == "tall":
                height_value = 0.8
        
        # Process shot types (0-5 scale from the API)
        for shot_id, count in api_data.get("shots", {}).items():
            shot_type = self.shot_mapping.get(shot_id)
            if shot_type in self.shot_type_cols:
                # Scale the count to match NBA data distribution using percentiles
                if self.percentiles and shot_type in self.percentiles:
                    percentile_key = "Low"  # Default for count=0
                    if count == 1:
                        percentile_key = "Low"
                    elif count == 2:
                        percentile_key = "Below Average"
                    elif count == 3:
                        percentile_key = "Average"
                    elif count == 4:
                        percentile_key = "Above Average"
                    elif count == 5:
                        percentile_key = "High"
                    
                    # Get the scaled value from percentiles
                    shot_features[shot_type] = float(self.percentiles[shot_type][percentile_key])
                    
                    # Set weight based on preference (higher preference = higher weight)
                    feature_weights[shot_type] = count / 5.0 if count > 0 else 0.1
                else:
                    # Simple linear scaling if percentiles not available
                    shot_features[shot_type] = float(count * 20)  # Scale 0-5 to 0-100
                    feature_weights[shot_type] = count / 5.0 if count > 0 else 0.1
        
        # Process court zones (none, low, medium, high from the API)
        for zone_id, intensity in api_data.get("zones", {}).items():
            # Clean zone ID by removing the "nba-" prefix if present
            clean_zone_id = zone_id.replace("nba-", "")
            
            # Map to feature column
            zone_name = self.zone_mapping.get(clean_zone_id)
            if zone_name in self.zone_cols:
                # Scale intensity using percentiles
                if self.percentiles and zone_name in self.percentiles:
                    percentile_key = self.intensity_mapping.get(intensity, "Low")
                    zone_features[zone_name] = float(self.percentiles[zone_name][percentile_key])
                    
                    # Set weight based on intensity (higher intensity = higher weight)
                    intensity_weight = {
                        "none": 0.1,  # Almost ignore
                        "low": 0.4,   # Give some importance
                        "medium": 0.7, # Moderate importance
                        "high": 1.0    # High importance
                    }
                    feature_weights[zone_name] = intensity_weight.get(intensity, 0.1)
                else:
                    # Simple linear scaling if percentiles not available
                    intensity_value = {
                        "none": 0, 
                        "low": 33, 
                        "medium": 67, 
                        "high": 100
                    }.get(intensity, 0)
                    zone_features[zone_name] = float(intensity_value)
                    
                    # Set weight based on intensity
                    intensity_weight = {
                        "none": 0.1,
                        "low": 0.4,
                        "medium": 0.7,
                        "high": 1.0
                    }
                    feature_weights[zone_name] = intensity_weight.get(intensity, 0.1)
        
        # Create DataFrames with feature column names for proper transformation
        shot_features_df = pd.DataFrame([shot_features], columns=self.shot_type_cols)
        zone_features_df = pd.DataFrame([zone_features], columns=self.zone_cols)
        
        # Normalize using the fitted scalers
        normalized_shot_features = self.shot_type_scaler.transform(shot_features_df).flatten()
        normalized_zone_features = self.zone_scaler.transform(zone_features_df).flatten()
        
        # Apply PCA if enabled
        if self.use_pca:
            # Transform shot features with PCA
            shot_pca_features = self.shot_type_pca.transform(normalized_shot_features.reshape(1, -1))
            # Transform zone features with PCA
            zone_pca_features = self.zone_pca.transform(normalized_zone_features.reshape(1, -1))
            
            # Use only the optimal number of components
            shot_pca_features = shot_pca_features[0, :self.n_shot_components]
            zone_pca_features = zone_pca_features[0, :self.n_zone_components]
            
            # Map feature weights to PCA space
            pca_shot_weights = self._map_feature_weights_to_pca(feature_weights, 'shot_types')
            pca_zone_weights = self._map_feature_weights_to_pca(feature_weights, 'zones')
            
            # Create combined feature vector with PCA components
            combined_features = np.concatenate([shot_pca_features, zone_pca_features])
            
            # Create a Series with appropriate index names for PCA components
            shot_indices = [f"shot_pc_{i+1}" for i in range(self.n_shot_components)]
            zone_indices = [f"zone_pc_{i+1}" for i in range(self.n_zone_components)]
            
            # Combine all features into a single Series with named indices
            all_features = pd.Series(combined_features, index=shot_indices + zone_indices)
            
            # Create a dictionary of PCA feature weights
            pca_feature_weights = {}
            for i, weight in enumerate(pca_shot_weights):
                pca_feature_weights[f"shot_pc_{i+1}"] = weight
            for i, weight in enumerate(pca_zone_weights):
                pca_feature_weights[f"zone_pc_{i+1}"] = weight
            
            # Add height as a separate value (not in the feature vector for KNN but used in similarity)
            all_features = pd.concat([all_features, pd.Series([height_value], index=['height'])])
            
            # Return PCA features and weights
            return all_features, pca_feature_weights
        else:
            # Without PCA, use the normalized features directly
            # Combine all normalized features into a single Series
            all_features = pd.Series(
                np.concatenate([normalized_shot_features, normalized_zone_features]),
                index=self.shot_type_cols + self.zone_cols
            )
            
            # Add height as a separate value (not in the feature vector for KNN, but used in similarity)
            all_features = pd.concat([all_features, pd.Series([height_value], index=['height'])])
            
            return all_features, feature_weights
    
    def calculate_weighted_similarity(self, 
                                     user_features: pd.Series, 
                                     nba_player_idx: int, 
                                     feature_weights: Dict[str, float]) -> float:
        """
        Calculate weighted similarity between user and NBA player.
        
        Parameters:
        user_features (pd.Series): User's feature vector
        nba_player_idx (int): Index of the NBA player in the original dataset
        feature_weights (Dict[str, float]): Weights for each feature
        
        Returns:
        float: Weighted similarity score (0-1)
        """
        if self.use_pca:
            # In PCA mode, we have precalculated PCA features
            # Get user's PCA features (excluding height)
            shot_pc_indices = [col for col in user_features.index if col.startswith('shot_pc_')]
            zone_pc_indices = [col for col in user_features.index if col.startswith('zone_pc_')]
            
            user_shot_features = user_features[shot_pc_indices].values
            user_zone_features = user_features[zone_pc_indices].values
            user_height = user_features['height']
            
            # Get player's PCA features
            player_shot_features = self.shot_type_pca_features[nba_player_idx, :self.n_shot_components]
            player_zone_features = self.zone_pca_features[nba_player_idx, :self.n_zone_components]
            
            # Calculate weighted distances for each category using feature weights
            shot_distances = []
            for i, pc_name in enumerate(shot_pc_indices):
                weight = feature_weights.get(pc_name, 0.1)
                # Higher weight = more impact on distance
                distance = (user_shot_features[i] - player_shot_features[i])**2 * weight
                shot_distances.append(distance)
            
            zone_distances = []
            for i, pc_name in enumerate(zone_pc_indices):
                weight = feature_weights.get(pc_name, 0.1)
                # Higher weight = more impact on distance
                distance = (user_zone_features[i] - player_zone_features[i])**2 * weight
                zone_distances.append(distance)
        else:
            # Without PCA, compare features in the original space
            # Get user's normalized features (excluding height)
            user_shot_features = user_features[self.shot_type_cols].values
            user_zone_features = user_features[self.zone_cols].values
            user_height = user_features['height']
            
            # Get player's normalized features
            player_shot_features = self.shot_type_features_scaled[nba_player_idx, :]
            player_zone_features = self.zone_features_scaled[nba_player_idx, :]
            
            # Calculate weighted distances for each category using feature weights
            shot_distances = []
            for i, col in enumerate(self.shot_type_cols):
                weight = feature_weights.get(col, 0.1)
                # Higher weight = more impact on distance
                distance = (user_shot_features[i] - player_shot_features[i])**2 * weight
                shot_distances.append(distance)
            
            zone_distances = []
            for i, col in enumerate(self.zone_cols):
                weight = feature_weights.get(col, 0.1)
                # Higher weight = more impact on distance
                distance = (user_zone_features[i] - player_zone_features[i])**2 * weight
                zone_distances.append(distance)
        
        # Get player height and calculate height distance
        player_name = self.player_data.iloc[nba_player_idx]['PLAYER_NAME']
        player_height = self.player_heights.get(player_name, 0.5)
        height_distance = (user_height - player_height)**2
        
        # Apply category weights for balanced contribution
        avg_shot_distance = np.sqrt(np.mean(shot_distances)) if shot_distances else 0
        avg_zone_distance = np.sqrt(np.mean(zone_distances)) if zone_distances else 0
        height_distance = np.sqrt(height_distance)  # Convert back to linear scale after squaring
        
        # Calculate weighted distance with category weights
        weighted_distance = (
            avg_shot_distance * self.category_weights['shot_types'] +
            avg_zone_distance * self.category_weights['zones'] +
            height_distance * self.category_weights['height']
        )
        
        # Apply non-linear transformation to create more spread in similarity scores
        # Use an exponential decay function: similarity = e^(-k*distance)
        k = 3.0  # Tuning parameter - higher values create more spread
        similarity = np.exp(-k * weighted_distance)
        
        # Scale the similarity for better discrimination
        scaled_similarity = self._scale_similarity(similarity)
        
        return scaled_similarity
    
    def _scale_similarity(self, raw_similarity: float) -> float:
        """
        Apply a scaling function to create more distinction between similarity scores.
        
        Parameters:
        raw_similarity (float): Raw similarity score
        
        Returns:
        float: Scaled similarity score
        """
        # Parameters for sigmoid scaling function
        midpoint = 0.8  # Center point of the sigmoid (scores above this are "good matches")
        steepness = 10.0  # Controls how quickly scores transition from low to high
        
        # Apply sigmoid transformation
        if raw_similarity > 0.999:  # Perfect match
            return 1.0
        
        # Sigmoid function: 1/(1+e^(-steepness*(x-midpoint)))
        scaled = 1.0 / (1.0 + np.exp(-steepness * (raw_similarity - midpoint)))
        
        # Rescale to 0-1 range
        min_value = 1.0 / (1.0 + np.exp(steepness * midpoint))
        max_value = 1.0 / (1.0 + np.exp(-steepness * (1.0 - midpoint)))
        scaled_normalized = (scaled - min_value) / (max_value - min_value)
        
        return max(0.0, min(1.0, scaled_normalized))
    
    def _map_pca_to_original_features(self, pca_vector, feature_type):
        """
        Map a vector from PCA space back to the original feature space.
        Used for interpretability of PCA-based matches.
        
        Parameters:
        pca_vector (np.ndarray): Vector in PCA space
        feature_type (str): 'shot_types' or 'zones'
        
        Returns:
        pd.Series: Approximate representation in original feature space
        """
        if not self.use_pca:
            return None
        
        if feature_type == 'shot_types':
            # Reconstruct using the principal components and their loadings
            # Note: This is an approximation since we only use a subset of components
            reconstructed = pca_vector @ self.shot_type_loadings
            # Return as a Series with original feature names
            return pd.Series(reconstructed, index=self.shot_type_cols)
        
        elif feature_type == 'zones':
            # Reconstruct using the principal components and their loadings
            reconstructed = pca_vector @ self.zone_loadings
            # Return as a Series with original feature names
            return pd.Series(reconstructed, index=self.zone_cols)
        
        return None
    
    def find_similar_players(self, user_features: pd.Series, feature_weights: Dict[str, float], api_data: Dict, n_matches: int = 5) -> Dict:
        """
        Find NBA players most similar to the given feature vector using weighted KNN.
        
        Parameters:
        user_features (pd.Series): Preprocessed feature vector
        feature_weights (Dict[str, float]): Weights for each feature
        api_data (Dict): Original API request data
        n_matches (int): Number of similar players to return
        
        Returns:
        dict: Similar players with their similarity scores, shot stats, and per-game NBA stats
        """
        # Calculate custom weighted similarity for all players
        similarities = []
        
        for idx in range(len(self.player_data)):
            similarity = self.calculate_weighted_similarity(user_features, idx, feature_weights)
            similarities.append((idx, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top n players
        top_matches = similarities[:n_matches]
        
        # Format results
        matches = []
        for i, (idx, similarity) in enumerate(top_matches):
            player = self.player_data.iloc[idx]
            player_name = player["PLAYER_NAME"]
            
            # Format player information
            match = {
                "player_name": player_name,
                "similarity_score": float(similarity),
                "similarity_percentage": f"{float(similarity)*100:.1f}%",
                "stats": {
                    "total_made_shots": int(player["TOTAL_MADE_SHOTS"]),
                    "total_attempted_shots": int(player["TOTAL_ATTEMPTED_SHOTS"]),
                    "total_missed_shots": int(player["TOTAL_MISSED_SHOTS"])
                },
                "per_game_stats": {}
            }
            
            # Add image URL for player
            image_url = self._get_player_image_url(player_name)
            match["image_url"] = image_url
            
            # Add per-game NBA stats if they exist
            for stat in self.per_game_stats:
                if stat in player and not pd.isna(player[stat]):
                    # Handle percentages with special formatting
                    if '%' in stat:
                        match["per_game_stats"][stat] = f"{player[stat]:.3f}"
                    else:
                        match["per_game_stats"][stat] = player[stat]
                else:
                    match["per_game_stats"][stat] = None
            
            # Add shot type stats
            for shot_type in self.shot_type_cols:
                if shot_type in player:
                    match["stats"][shot_type.lower()] = int(player[shot_type])
            
            # Add simplified court zone stats using a more readable format
            zone_groups = {
                "paint": ["In The Paint (Non-RA) - Center", "Restricted Area - Center"],
                "mid_range": ["Mid-Range - Center", "Mid-Range - Left Side", "Mid-Range - Right Side", 
                             "Mid-Range - Left Side Center", "Mid-Range - Right Side Center"],
                "three_point": ["Left Corner 3 - Left Side", "Right Corner 3 - Right Side",
                               "Above the Break 3 - Left Side Center", "Above the Break 3 - Right Side Center",
                               "Above the Break 3 - Center"]
            }
            
            zone_stats = {}
            for group_name, zones in zone_groups.items():
                zone_stats[group_name] = sum(int(player[zone]) for zone in zones if zone in player)
            
            match["stats"]["zones"] = zone_stats
            
            # Add contribution factors to explain the match
            match["contribution_factors"] = self._calculate_contribution_factors(
                user_features, idx, feature_weights, api_data)
            
            matches.append(match)
        
        return {"matches": matches}
    
    def _calculate_contribution_factors(self, 
                                       user_features: pd.Series, 
                                       player_idx: int,
                                       feature_weights: Dict[str, float],
                                       api_data: Dict) -> Dict:
        """
        Calculate the factors that contributed most to the match.
        
        Parameters:
        user_features (pd.Series): User's feature vector
        player_idx (int): Index of player in the dataset
        feature_weights (Dict[str, float]): Weights for each feature
        api_data (Dict): Original API request data
        
        Returns:
        Dict: Contribution factors explaining the match
        """
        factors = {}
        player = self.player_data.iloc[player_idx]
        
        if self.use_pca:
            # For PCA mode, we need to map from PCA space back to original features
            # Extract PCA components from user features
            shot_pc_indices = [col for col in user_features.index if col.startswith('shot_pc_')]
            zone_pc_indices = [col for col in user_features.index if col.startswith('zone_pc_')]
            
            user_shot_pca = user_features[shot_pc_indices].values
            user_zone_pca = user_features[zone_pc_indices].values
            
            # Get player's PCA values
            player_shot_pca = self.shot_type_pca_features[player_idx, :self.n_shot_components]
            player_zone_pca = self.zone_pca_features[player_idx, :self.n_zone_components]
            
            # Map PCA values back to original feature space for interpretability
            user_shot_original = self._map_pca_to_original_features(user_shot_pca, 'shot_types')
            user_zone_original = self._map_pca_to_original_features(user_zone_pca, 'zones')
            
            player_shot_original = self._map_pca_to_original_features(player_shot_pca, 'shot_types')
            player_zone_original = self._map_pca_to_original_features(player_zone_pca, 'zones')
            
            # Calculate similarities in original feature space for interpretability
            shot_similarities = {}
            for shot_type in self.shot_type_cols:
                # Only include features with meaningful weight in original space
                orig_weight = 0.1  # Default weight
                for shot_id, mapped_type in self.shot_mapping.items():
                    if mapped_type == shot_type and shot_id in api_data.get('shots', {}):
                        orig_weight = api_data['shots'][shot_id] / 5.0
                
                if orig_weight > 0.2:
                    # Calculate similarity for this shot type in original space
                    user_value = user_shot_original[shot_type]
                    player_value = player_shot_original[shot_type]
                    similarity = 1.0 - abs(user_value - player_value)
                    shot_similarities[shot_type] = similarity * orig_weight  # Weight by user preference
            
            # Get top 3 matching shot types
            top_shots = sorted(shot_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
            factors["top_shot_matches"] = [{"shot_type": shot, "similarity": sim} for shot, sim in top_shots]
            
            # Calculate similarities for zones in original feature space
            zone_similarities = {}
            for zone in self.zone_cols:
                # Only include zones with meaningful weight in original space
                orig_weight = 0.1  # Default weight
                for zone_id, mapped_zone in self.zone_mapping.items():
                    if mapped_zone == zone and zone_id in api_data.get('zones', {}):
                        intensity = api_data['zones'][zone_id]
                        intensity_weight = {
                            "none": 0.1, "low": 0.4, "medium": 0.7, "high": 1.0
                        }.get(intensity, 0.1)
                        orig_weight = intensity_weight
                
                if orig_weight > 0.2:
                    # Calculate similarity for this zone in original space
                    user_value = user_zone_original[zone]
                    player_value = player_zone_original[zone]
                    similarity = 1.0 - abs(user_value - player_value)
                    zone_similarities[zone] = similarity * orig_weight  # Weight by user preference
            
            # Get top 3 matching zones
            top_zones = sorted(zone_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
            factors["top_zone_matches"] = [{"zone": zone, "similarity": sim} for zone, sim in top_zones]
        else:
            # Without PCA, calculate in original feature space
            shot_similarities = {}
            for shot_type in self.shot_type_cols:
                if shot_type in player and feature_weights.get(shot_type, 0) > 0.2:
                    # Get values
                    user_value = user_features[shot_type]
                    player_value = self.shot_type_scaler.transform(
                        [[player[shot_type]]] if shot_type in player else [[0]]
                    )[0][0]
                    
                    # Calculate similarity
                    similarity = 1.0 - abs(user_value - player_value)
                    shot_similarities[shot_type] = similarity
            
            # Get top 3 matching shot types
            top_shots = sorted(shot_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
            factors["top_shot_matches"] = [{"shot_type": shot, "similarity": sim} for shot, sim in top_shots]
            
            # Calculate similarities for zones
            zone_similarities = {}
            for zone in self.zone_cols:
                if zone in player and feature_weights.get(zone, 0) > 0.2:
                    # Get values
                    user_value = user_features[zone]
                    player_value = self.zone_scaler.transform(
                        [[player[zone]]] if zone in player else [[0]]
                    )[0][0]
                    
                    # Calculate similarity
                    similarity = 1.0 - abs(user_value - player_value)
                    zone_similarities[zone] = similarity
            
            # Get top 3 matching zones
            top_zones = sorted(zone_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
            factors["top_zone_matches"] = [{"zone": zone, "similarity": sim} for zone, sim in top_zones]
        
        # Add height match
        player_name = player['PLAYER_NAME']
        player_height = self.player_heights.get(player_name, 0.5)
        user_height = user_features['height']
        height_similarity = 1.0 - abs(user_height - player_height)
        factors["height_match"] = {"similarity": height_similarity}
        
        # Calculate contribution percentages
        shot_contribution = sum(s[1] for s in top_shots) * self.category_weights['shot_types'] if top_shots else 0
        zone_contribution = sum(z[1] for z in top_zones) * self.category_weights['zones'] if top_zones else 0
        height_contribution = height_similarity * self.category_weights['height']
        
        total_contribution = shot_contribution + zone_contribution + height_contribution
        
        if total_contribution > 0:
            factors["contribution_percentages"] = {
                "shot_types": shot_contribution / total_contribution * 100,
                "zones": zone_contribution / total_contribution * 100,
                "height": height_contribution / total_contribution * 100
            }
        
        return factors
    
    def match_player(self, api_data: Dict, n_matches: int = 5) -> Dict:
        """
        Match a player based on API data with PCA dimensionality reduction and improved weighting.
        
        Parameters:
        api_data (dict): API request payload with player height, shots, and zones
        n_matches (int): Number of matches to return
        
        Returns:
        dict: Matching results with similar NBA players and their per-game stats
        """
        try:
            # Debug the input data
            print(f"Processing input data: {json.dumps(api_data, indent=2)}")
            
            # Check player data
            print(f"Player data contains {len(self.player_data)} entries")
            print(f"Sample player names: {', '.join(self.player_data['PLAYER_NAME'].head(5).tolist())}")
            
            # Preprocess input and get feature weights
            user_features, feature_weights = self.preprocess_api_input(api_data)
            
            # Log user features for debugging
            print(f"Processed user features:\n{user_features}")
            print(f"Feature weights: {json.dumps({k: round(v, 2) for k, v in feature_weights.items()}, indent=2)}")
            
            # Find similar players using weighted similarity - pass api_data
            result = self.find_similar_players(user_features, feature_weights, api_data, n_matches)

            # Convert NumPy types to native Python types before returning
            result = convert_numpy_types(result)

            
            # Log match results count
            print(f"Found {len(result.get('matches', []))} matches")
            
            return result
        
        except Exception as e:
            print(f"Error in match_player: {str(e)}")
            import traceback
            traceback.print_exc()
            error_result = {
                "error": str(e),
                "matches": [],
                "debug_info": {
                    "player_data_shape": self.player_data.shape if hasattr(self, 'player_data') else None,
                    "api_data": api_data
                }
            }
        
            return convert_numpy_types(error_result)  # Also convert error response


# Helper function to create percentile dictionary from dataframe
def create_percentile_dict(df: pd.DataFrame) -> Dict:
    """
    Create a dictionary of percentiles for each feature column.
    
    Parameters:
    df (pd.DataFrame): DataFrame with feature columns
    
    Returns:
    dict: Dictionary of percentiles for each feature
    """
    percentile_dict = {}
    
    # Get feature columns (all columns except player info and per-game stats)
    feature_cols = [col for col in df.columns 
                   if col not in ["PLAYER_NAME", "TOTAL_MADE_SHOTS", 
                                 "TOTAL_ATTEMPTED_SHOTS", "TOTAL_MISSED_SHOTS"] and
                     col not in ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 
                                'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 
                                'STL', 'BLK', 'TOV', 'PF', 'PTS']]
    
    # Calculate percentiles for each feature
    for column in feature_cols:
        percentiles = {
            "Low": float(df[column].quantile(0.1)),
            "Below Average": float(df[column].quantile(0.3)),
            "Average": float(df[column].quantile(0.5)),
            "Above Average": float(df[column].quantile(0.7)),
            "High": float(df[column].quantile(0.9))
        }
        
        # Round values to make them more readable
        for percentile in percentiles:
            percentiles[percentile] = round(percentiles[percentile], 2)
        
        percentile_dict[column] = percentiles
    
    return percentile_dict


# FastAPI implementation with proper CORS support
def create_api():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    from typing import Dict, Optional
    import traceback
    import os
    
    app = FastAPI(title="Enhanced NBA Player Shot Profile Matcher with PCA")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://qiwens-dapper-site.webflow.io", "http://localhost:3000", "*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Mount the static files directory for player images
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Player Pictures")
    if os.path.exists(static_dir):
        app.mount("/images", StaticFiles(directory=static_dir), name="player_images")
    else:
        print(f"Warning: Player images directory not found at {static_dir}")
    
    class PlayerHeight(BaseModel):
        height: str = Field(..., description="Player height (short, average, tall)")
    
    class PlayerData(BaseModel):
        player: PlayerHeight
        shots: Dict[str, int] = Field(default_factory=dict, description="Shot type preferences (0-5 scale)")
        zones: Dict[str, str] = Field(default_factory=dict, description="Court zone preferences (none, low, medium, high)")
    
    # Initialize the matcher (loaded once when the server starts)
    player_matcher = None
    
    @app.on_event("startup")
    async def startup_event():
        nonlocal player_matcher
        try:
            # Try to load the data - with verbose error reporting
            print("Starting matcher initialization...")
            
            import os
            
            # Get data directory from environment variable or use current directory
            data_dir = os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
            
            # Construct paths to data files
            data_path = os.path.join(data_dir, "complete_df.csv")
            percentile_path = os.path.join(data_dir, "percentiles.json")
            
            # Check if data files exist
            if not os.path.exists(data_path):
                print(f"ERROR: Data file not found: {data_path}")
                # Try to find alternative files
                csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                print(f"Available CSV files: {csv_files}")
                
                if csv_files:
                    alternative_path = os.path.join(data_dir, csv_files[0])
                    print(f"Using alternative data file: {alternative_path}")
                    data_path = alternative_path
            else:
                print(f"Found data file: {data_path}")
            
            # Initialize the matcher with PCA
            player_matcher = EnhancedNBAPlayerMatcher(
                data_path=data_path,
                percentile_path=percentile_path if os.path.exists(percentile_path) else None,
                pca_variance_threshold=0.95,
                use_pca=True
            )
            
            # Log success
            print(f"Enhanced NBA Player Matcher initialized successfully with {len(player_matcher.player_data)} players")
            
            # Print PCA information
            if player_matcher.use_pca:
                print(f"PCA: Shot types using {player_matcher.n_shot_components} components " +
                      f"(variance: {player_matcher.shot_type_pca.explained_variance_ratio_[:player_matcher.n_shot_components].sum()*100:.2f}%)")
                print(f"PCA: Zones using {player_matcher.n_zone_components} components " +
                      f"(variance: {player_matcher.zone_pca.explained_variance_ratio_[:player_matcher.n_zone_components].sum()*100:.2f}%)")
                
        except Exception as e:
            print(f"ERROR initializing Enhanced NBA Player Matcher: {str(e)}")
            traceback.print_exc()
    
    # Simple health check endpoint
    @app.get("/")
    async def root():
        return {
            "status": "Enhanced API with PCA is running",
            "matcher_initialized": player_matcher is not None,
            "player_count": len(player_matcher.player_data) if player_matcher is not None else 0,
            "pca_enabled": player_matcher.use_pca if player_matcher is not None else False,
            "pca_components": {
                "shot_types": player_matcher.n_shot_components if player_matcher is not None and player_matcher.use_pca else None,
                "zones": player_matcher.n_zone_components if player_matcher is not None and player_matcher.use_pca else None
            }
        }
    
    @app.post("/api/match-player")
    async def match_player(data: PlayerData):
        """
        Match a player based on their shot profile with PCA dimensionality reduction and improved weighting.
        
        This endpoint takes a player's profile (height, shot types, and court zones),
        applies PCA dimensionality reduction, and returns the NBA players who most 
        closely match this profile using an enhanced weighted algorithm.
        """
        if player_matcher is None:
            raise HTTPException(status_code=500, detail="NBA Player Matcher not initialized")
        
        try:
            # Use model_dump() instead of dict() for Pydantic v2
            api_data = data.model_dump()
            
            # Log the incoming request
            print(f"Received match request: {json.dumps(api_data, indent=2)}")
            
            # Process the request
            result = player_matcher.match_player(api_data)
            
            # Convert NumPy types to Python native types
            result = convert_numpy_types(result)
            
            # Log success
            match_count = len(result.get("matches", []))
            print(f"Successfully matched player with {match_count} results")
            
            return result
            
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error matching player: {str(e)}")
            print(error_trace)
            
            # Return a more helpful error response with converted types
            error_content = convert_numpy_types({
                "detail": f"Error matching player: {str(e)}",
                "error_trace": error_trace,
                "matches": []
            })
            
            return JSONResponse(
                status_code=500,
                content=error_content
            )


# Example usage
if __name__ == "__main__":
    # Import necessary modules
    import os
    import uvicorn
    
    # Create the API
    app = create_api()
    
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Start the API server
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")