import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SituationAnalyzer:
    def __init__(self):
        self.previous_analysis = None
        
    def analyze(self, objects):
        """Analyze the overall situation based on detected objects"""
        try:
            # Count objects by type
            object_counts = {}
            for obj in objects:
                obj_class = obj['class']
                object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
            
            # Get people count
            people_count = object_counts.get('person', 0)
            
            # Analyze activity level
            activity_level = self._calculate_activity_level(objects)
            
            # Generate situation description
            description = self._generate_description(object_counts, activity_level)
            
            # Determine environment type
            environment = self._determine_environment(object_counts)
            
            analysis = {
                'description': description,
                'environment': environment,
                'activity_level': activity_level,
                'people_count': people_count,
                'object_counts': object_counts,
                'primary_objects': self._get_primary_objects(object_counts),
                'timestamp': datetime.now().isoformat()
            }
            
            self.previous_analysis = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Situation analysis failed: {str(e)}")
            return self._get_fallback_analysis()

    def _calculate_activity_level(self, objects):
        """Calculate activity level based on objects and movement"""
        people_count = len([obj for obj in objects if obj['class'] == 'person'])
        vehicle_count = len([obj for obj in objects if obj['class'] in ['car', 'vehicle', 'bicycle', 'motorcycle']])
        
        if people_count > 5 or vehicle_count > 3:
            return "high"
        elif people_count > 2 or vehicle_count > 1:
            return "medium"
        else:
            return "low"

    def _generate_description(self, object_counts, activity_level):
        """Generate natural language description of the situation"""
        people = object_counts.get('person', 0)
        vehicles = object_counts.get('car', 0) + object_counts.get('vehicle', 0)
        animals = object_counts.get('dog', 0) + object_counts.get('cat', 0)
        
        descriptions = []
        
        if people == 0:
            descriptions.append("The area appears to be empty")
        elif people == 1:
            descriptions.append("There is one person present")
        else:
            descriptions.append(f"There are {people} people in the area")
            
        if vehicles > 0:
            descriptions.append(f"{vehicles} vehicle(s) detected")
            
        if animals > 0:
            descriptions.append(f"{animals} animal(s) spotted")
            
        # Add activity context
        if activity_level == "high":
            descriptions.append("The scene shows high activity")
        elif activity_level == "medium":
            descriptions.append("Moderate activity observed")
        else:
            descriptions.append("The environment is calm")
            
        return ". ".join(descriptions) + "."

    def _determine_environment(self, object_counts):
        """Determine the type of environment"""
        indoor_objects = ['chair', 'table', 'furniture', 'computer', 'tv']
        outdoor_objects = ['car', 'tree', 'sky', 'road']
        
        indoor_score = sum(object_counts.get(obj, 0) for obj in indoor_objects)
        outdoor_score = sum(object_counts.get(obj, 0) for obj in outdoor_objects)
        
        if indoor_score > outdoor_score:
            return "indoor"
        elif outdoor_score > indoor_score:
            return "outdoor"
        else:
            return "unknown"

    def _get_primary_objects(self, object_counts):
        """Get the most prominent objects"""
        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        return [obj[0] for obj in sorted_objects[:3]]

    def _get_fallback_analysis(self):
        """Return fallback analysis when primary analysis fails"""
        return {
            'description': 'Unable to analyze scene. Please try again.',
            'environment': 'unknown',
            'activity_level': 'unknown',
            'people_count': 0,
            'object_counts': {},
            'primary_objects': [],
            'timestamp': datetime.now().isoformat()
        }
