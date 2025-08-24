#!/usr/bin/env python3
"""
Satellite Image Change Detector
Compares 2 satellite images and explains changes in human-like language using YOLO, OpenCV, and Gemini AI
"""

import cv2
import numpy as np
import os
import argparse
import requests
from PIL import Image
import google.generativeai as genai
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime

class SatelliteChangeDetector:
    def __init__(self, gemini_api_key):
        """Initialize the change detector with Gemini API key"""
        self.gemini_api_key = gemini_api_key
        self.setup_gemini()
        self.setup_yolo()
        
    def setup_gemini(self):
        """Setup Google Gemini AI"""
        try:
            genai.configure(api_key=self.gemini_api_key)
            
            # Try different Gemini models in order of preference
            models_to_try = [
                'gemini-1.5-pro',           # Latest model
                'gemini-1.5-flash',         # Fast model
                'gemini-1.0-pro',           # Previous version
                'gemini-pro'                 # Fallback
            ]
            
            self.gemini_model = None
            for model_name in models_to_try:
                try:
                    print(f"   Trying model: {model_name}...")
                    self.gemini_model = genai.GenerativeModel(model_name)
                    # Test if model supports vision
                    test_response = self.gemini_model.generate_content("Hello")
                    print(f"✅ Gemini AI initialized successfully with {model_name}")
                    break
                except Exception as model_error:
                    print(f"   ⚠️ {model_name} failed: {model_error}")
                    continue
            
            if self.gemini_model is None:
                print("❌ All Gemini models failed to initialize")
                print("   This might be due to:")
                print("   - Invalid API key")
                print("   - Network connectivity issues")
                print("   - API quota exceeded")
                print("   - Model availability issues")
                
                # Try to list available models for debugging
                try:
                    print("\n🔍 Checking available models...")
                    models = genai.list_models()
                    print("   Available models:")
                    for model in models:
                        print(f"     - {model.name}")
                except Exception as list_error:
                    print(f"   Could not list models: {list_error}")
                
                print("\n💡 The system will continue with text-based analysis.")
                
        except Exception as e:
            print(f"❌ Failed to initialize Gemini AI: {e}")
            self.gemini_model = None
    
    def setup_yolo(self):
        """Setup YOLO model for object detection"""
        try:
            # Use YOLO for detecting objects in satellite images
            self.yolo_model = YOLO('yolov8n.pt')  # Use nano model for speed
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            self.yolo_model = None
    
    def load_image(self, image_path):
        """Load and validate satellite image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Resize for consistent processing (maintain aspect ratio)
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        return image
    
    def detect_objects_yolo(self, image):
        """Detect objects in image using YOLO"""
        if self.yolo_model is None:
            return []
        
        try:
            # Convert BGR to RGB for YOLO
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection
            results = self.yolo_model(rgb_image)
            
            # Extract detected objects
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class': class_name,
                            'class_id': class_id
                        })
            
            return detections
        except Exception as e:
            print(f"⚠️ YOLO detection failed: {e}")
            return []
    
    def detect_changes_opencv(self, image1, image2):
        """Detect changes between two images using OpenCV"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Ensure same size
            if gray1.shape != gray2.shape:
                gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Apply threshold to get binary change map
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours of changed regions
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter significant changes
            significant_changes = []
            min_area = 100  # Minimum area to consider significant
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    significant_changes.append({
                        'bbox': [x, y, x + w, y + h],
                        'area': area,
                        'center': (x + w//2, y + h//2)
                    })
            
            return cleaned, significant_changes
            
        except Exception as e:
            print(f"❌ OpenCV change detection failed: {e}")
            return None, []
    
    def create_change_visualization(self, image1, image2, change_map, changes, output_path=None):
        """Create visualization of detected changes"""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Satellite Image Change Detection Results', fontsize=16)
            
            # Original images
            rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            rgb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            
            axes[0, 0].imshow(rgb1)
            axes[0, 0].set_title('Image 1 (Before)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(rgb2)
            axes[0, 1].set_title('Image 2 (After)')
            axes[0, 1].axis('off')
            
            # Change map
            axes[1, 0].imshow(change_map, cmap='hot')
            axes[1, 0].set_title('Change Detection Map')
            axes[1, 0].axis('off')
            
            # Overlay changes on second image
            overlay = rgb2.copy()
            for change in changes:
                x1, y1, x2, y2 = change['bbox']
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(overlay, f"Change {change['area']:.0f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('Detected Changes Overlay')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"📊 Visualization saved to: {output_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
    
    def analyze_changes_with_gemini(self, image1, image2, change_map, changes, yolo_detections1, yolo_detections2):
        """Analyze changes using Gemini AI for human-like explanation"""
        if self.gemini_model is None:
            return self.generate_fallback_analysis(changes, yolo_detections1, yolo_detections2)
        
        try:
            # Prepare images for Gemini
            pil_image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            pil_image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
            
            # Convert change map to PIL image
            change_pil = Image.fromarray(change_map)
            
            # Create comprehensive prompt for more human-like analysis
            prompt = f"""
            You are a satellite imagery expert explaining changes to a non-technical audience.
            
            Analyze these two satellite images and explain the changes in simple, human terms.
            
            CONTEXT:
            - Image 1: "Before" satellite image
            - Image 2: "After" satellite image  
            - Change Map: Binary image showing detected differences (white areas = changes)
            
            DETECTED OBJECTS (Image 1):
            {self._format_yolo_detections(yolo_detections1)}
            
            DETECTED OBJECTS (Image 2):
            {self._format_yolo_detections(yolo_detections2)}
            
            CHANGE STATISTICS:
            - Total change regions: {len(changes)}
            - Total change area: {sum(c['area'] for c in changes):.0f} pixels
            
            Please provide a clear, conversational explanation that includes:
            
            **What Changed?**
            Describe the visible differences in simple, everyday language. What would someone notice if they looked at both images?
            
            **Why Does This Matter?**
            Explain why these changes are significant. How do they affect the area or people?
            
            **What Caused These Changes?**
            Suggest possible reasons (construction, natural events, human activity, etc.)
            
            **Environmental Impact**
            How might these changes affect the environment, wildlife, or local ecosystem?
            
            **What Should Happen Next?**
            Give practical recommendations for monitoring, response, or further investigation.
            
            Write as if you're talking to a friend who's curious about satellite images.
            Use simple language, avoid technical jargon, and make it engaging and easy to understand.
            """
            
            # Generate analysis
            response = self.gemini_model.generate_content([prompt, pil_image1, pil_image2, change_pil])
            return response.text
            
        except Exception as e:
            print(f"⚠️ Gemini AI analysis failed: {e}")
            print("   Falling back to text-based analysis...")
            return self.generate_fallback_analysis(changes, yolo_detections1, yolo_detections2)
    
    def generate_fallback_analysis(self, changes, yolo_detections1, yolo_detections2):
        """Generate a fallback analysis when Gemini AI is not available"""
        analysis = []
        
        # What Changed
        analysis.append("**What Changed?**")
        if changes:
            total_area = sum(c['area'] for c in changes)
            analysis.append(f"I detected {len(changes)} significant change regions between the two satellite images.")
            analysis.append(f"The total area of changes is {total_area:,} pixels.")
            
            # Categorize changes
            small_changes = [c for c in changes if c['area'] < 500]
            medium_changes = [c for c in changes if 500 <= c['area'] < 2000]
            large_changes = [c for c in changes if c['area'] >= 2000]
            
            if large_changes:
                analysis.append(f"There are {len(large_changes)} large change areas, which could indicate major developments like new construction or significant land use changes.")
            if medium_changes:
                analysis.append(f"There are {len(medium_changes)} medium-sized changes, possibly representing infrastructure modifications or moderate environmental changes.")
            if small_changes:
                analysis.append(f"There are {len(small_changes)} small changes, which might be minor modifications or natural variations.")
        else:
            analysis.append("No significant changes were detected between the two images.")
        
        # Why It Matters
        analysis.append("\n**Why Does This Matter?**")
        if changes:
            analysis.append("These changes could indicate:")
            analysis.append("- Urban development and infrastructure expansion")
            analysis.append("- Environmental modifications or land use changes")
            analysis.append("- Natural events like seasonal variations or weather impacts")
            analysis.append("- Human activities such as construction or agriculture")
        else:
            analysis.append("The lack of significant changes suggests the area has remained relatively stable.")
        
        # Object Detection Insights
        if yolo_detections1 or yolo_detections2:
            analysis.append("\n**Object Detection Insights**")
            if yolo_detections1:
                obj_types1 = [det['class'] for det in yolo_detections1]
                analysis.append(f"In the 'before' image, I detected: {', '.join(set(obj_types1))}")
            if yolo_detections2:
                obj_types2 = [det['class'] for det in yolo_detections2]
                analysis.append(f"In the 'after' image, I detected: {', '.join(set(obj_types2))}")
            
            # Compare object counts
            if yolo_detections1 and yolo_detections2:
                if len(yolo_detections2) > len(yolo_detections1):
                    analysis.append("The increase in detected objects suggests development or increased activity in the area.")
                elif len(yolo_detections2) < len(yolo_detections1):
                    analysis.append("The decrease in detected objects might indicate clearing or removal of structures.")
        
        # Recommendations
        analysis.append("\n**What Should Happen Next?**")
        if changes:
            analysis.append("Based on the detected changes, I recommend:")
            analysis.append("- Further monitoring of the changed areas")
            analysis.append("- Investigation into the causes of significant changes")
            analysis.append("- Assessment of environmental impact")
            analysis.append("- Documentation for planning and regulatory purposes")
        else:
            analysis.append("Since no significant changes were detected, continue regular monitoring as part of standard procedures.")
        
        return "\n".join(analysis)
    
    def _format_yolo_detections(self, detections):
        """Format YOLO detections for the prompt"""
        if not detections:
            return "No objects detected"
        
        formatted = []
        for det in detections:
            formatted.append(f"- {det['class']} (confidence: {det['confidence']:.2f})")
        
        return "\n".join(formatted)
    
    def display_human_friendly_results(self, analysis, changes, yolo_detections1, yolo_detections2):
        """Display results in a human-friendly, well-formatted way"""
        print("\n" + "🌟" + "=" * 48 + "🌟")
        print("🛰️  SATELLITE CHANGE DETECTION RESULTS")
        print("🌟" + "=" * 48 + "🌟")
        
        # Summary Statistics
        print("\n📊 QUICK SUMMARY:")
        print("   • Total changes detected:", len(changes))
        total_area = sum(c['area'] for c in changes)
        print(f"   • Total change area: {total_area:,} pixels")
        
        if yolo_detections1 or yolo_detections2:
            print(f"   • Objects in 'before' image: {len(yolo_detections1)}")
            print(f"   • Objects in 'after' image: {len(yolo_detections2)}")
        
        # Change Details
        if changes:
            print("\n🔍 CHANGE DETAILS:")
            for i, change in enumerate(changes[:5], 1):  # Show first 5 changes
                area = change['area']
                x, y, w, h = change['bbox']
                print(f"   {i}. Change #{i}: {area:,} pixels at position ({x}, {y})")
            
            if len(changes) > 5:
                print(f"   ... and {len(changes) - 5} more changes")
        
        # Object Detection Summary
        if yolo_detections1 or yolo_detections2:
            print("\n🎯 OBJECT DETECTION SUMMARY:")
            
            if yolo_detections1:
                print("   📸 In 'Before' Image:")
                object_counts1 = {}
                for det in yolo_detections1:
                    obj_type = det['class']
                    object_counts1[obj_type] = object_counts1.get(obj_type, 0) + 1
                
                for obj_type, count in object_counts1.items():
                    print(f"      • {obj_type}: {count}")
            
            if yolo_detections2:
                print("   📸 In 'After' Image:")
                object_counts2 = {}
                for det in yolo_detections2:
                    obj_type = det['class']
                    object_counts2[obj_type] = object_counts2.get(obj_type, 0) + 1
                
                for obj_type, count in object_counts2.items():
                    print(f"      • {obj_type}: {count}")
        
        # AI Analysis
        print("\n🤖 AI ANALYSIS:")
        print("   " + "─" * 40)
        
        # Split analysis into sections for better readability
        analysis_lines = analysis.split('\n')
        current_section = ""
        
        for line in analysis_lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a section header
            if line.startswith('**') and line.endswith('**'):
                current_section = line.strip('*')
                print(f"\n   📌 {current_section}")
            elif line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.') or line.startswith('5.'):
                print(f"   {line}")
            elif line.startswith('-'):
                print(f"      {line}")
            else:
                # Regular paragraph
                if len(line) > 80:
                    # Wrap long lines
                    words = line.split()
                    current_line = "   "
                    for word in words:
                        if len(current_line + word) > 80:
                            print(current_line)
                            current_line = "   " + word + " "
                        else:
                            current_line += word + " "
                    if current_line.strip():
                        print(current_line)
                else:
                    print(f"   {line}")
        
        print("\n" + "🌟" + "=" * 48 + "🌟")
        print("💡 TIP: Check the 'output' folder for detailed reports and visualizations!")
        print("🌟" + "=" * 48 + "🌟")
    
    def show_quick_summary(self, changes, yolo_detections1, yolo_detections2):
        """Show a quick summary of the analysis results"""
        print("\n📋 QUICK SUMMARY:")
        print("   " + "─" * 40)
        
        # Change summary
        if changes:
            total_area = sum(c['area'] for c in changes)
            print(f"   🚨 Changes Detected: {len(changes)} regions")
            print(f"   📏 Total Change Area: {total_area:,} pixels")
            
            # Categorize changes by size
            small_changes = [c for c in changes if c['area'] < 500]
            medium_changes = [c for c in changes if 500 <= c['area'] < 2000]
            large_changes = [c for c in changes if c['area'] >= 2000]
            
            if small_changes:
                print(f"      • Small changes: {len(small_changes)}")
            if medium_changes:
                print(f"      • Medium changes: {len(medium_changes)}")
            if large_changes:
                print(f"      • Large changes: {len(large_changes)}")
        else:
            print("   ✅ No significant changes detected")
        
        # Object detection summary
        if yolo_detections1 or yolo_detections2:
            print(f"   🎯 Objects Detected: {len(yolo_detections1)} before, {len(yolo_detections2)} after")
            
            # Most common objects
            if yolo_detections1:
                obj_types1 = [det['class'] for det in yolo_detections1]
                if obj_types1:
                    most_common1 = max(set(obj_types1), key=obj_types1.count)
                    print(f"      • Most common in 'before': {most_common1}")
            
            if yolo_detections2:
                obj_types2 = [det['class'] for det in yolo_detections2]
                if obj_types2:
                    most_common2 = max(set(obj_types2), key=obj_types2.count)
                    print(f"      • Most common in 'after': {most_common2}")
        
        print("   " + "─" * 40)
        print()
    
    def process_images(self, image1_path, image2_path, output_dir="output"):
        """Main processing function"""
        print("🛰️  Satellite Image Change Detection")
        print("=" * 50)
        print("🚀 Starting analysis... This may take a few moments.")
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load images
            print("📸 Step 1/5: Loading satellite images...")
            image1 = self.load_image(image1_path)
            image2 = self.load_image(image2_path)
            print(f"   ✅ Loaded images: {image1.shape[1]}x{image1.shape[0]} and {image2.shape[1]}x{image2.shape[0]}")
            
            # Detect objects with YOLO
            print("\n🔍 Step 2/5: Detecting objects with YOLO...")
            print("   This may take a moment as YOLO analyzes the images...")
            yolo_detections1 = self.detect_objects_yolo(image1)
            yolo_detections2 = self.detect_objects_yolo(image2)
            print(f"   ✅ YOLO detected {len(yolo_detections1)} objects in Image 1, {len(yolo_detections2)} objects in Image 2")
            
            # Detect changes with OpenCV
            print("\n🔄 Step 3/5: Detecting changes with OpenCV...")
            change_map, changes = self.detect_changes_opencv(image1, image2)
            
            if change_map is not None:
                print(f"   ✅ Detected {len(changes)} significant change regions")
                
                # Create visualization
                print("\n📊 Step 4/5: Creating visualization...")
                viz_path = os.path.join(output_dir, "change_detection_visualization.png")
                self.create_change_visualization(image1, image2, change_map, changes, viz_path)
                print("   ✅ Visualization created and saved")
                
                # Analyze with Gemini AI
                print("\n🤖 Step 5/5: Analyzing changes with Gemini AI...")
                print("   Getting AI-powered insights...")
                analysis = self.analyze_changes_with_gemini(
                    image1, image2, change_map, changes, yolo_detections1, yolo_detections2
                )
                print("   ✅ AI analysis completed")
                
                # Save analysis
                print("\n💾 Saving results...")
                analysis_path = os.path.join(output_dir, "change_analysis.txt")
                with open(analysis_path, 'w') as f:
                    f.write("Satellite Image Change Analysis\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Image 1: {image1_path}\n")
                    f.write(f"Image 2: {image2_path}\n\n")
                    f.write(f"Detected Changes: {len(changes)}\n")
                    f.write(f"Total Change Area: {sum(c['area'] for c in changes):.0f} pixels\n\n")
                    f.write("AI Analysis:\n")
                    f.write("-" * 20 + "\n")
                    f.write(analysis)
                
                print(f"   ✅ Analysis saved to: {analysis_path}")
                
                print("\n" + "🎯" + "=" * 48 + "🎯")
                print("📋 ANALYSIS COMPLETE! Displaying results below...")
                print("🎯" + "=" * 48 + "🎯")
                
                # Show quick summary first
                self.show_quick_summary(changes, yolo_detections1, yolo_detections2)
                
                # Display detailed results in a more human-friendly way
                self.display_human_friendly_results(analysis, changes, yolo_detections1, yolo_detections2)
                
                return True
            else:
                print("❌ Failed to detect changes")
                return False
                
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Detect and analyze changes in satellite images using YOLO, OpenCV, and Gemini AI"
    )
    parser.add_argument("image1", help="Path to first satellite image (before)")
    parser.add_argument("image2", help="Path to second satellite image (after)")
    parser.add_argument("--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY environment variable)")
    
    args = parser.parse_args()
    
    # Check if image files exist
    if not os.path.exists(args.image1):
        print(f"❌ First image not found: {args.image1}")
        print("Please provide a valid path to your first satellite image.")
        sys.exit(1)
    
    if not os.path.exists(args.image2):
        print(f"❌ Second image not found: {args.image2}")
        print("Please provide a valid path to your second satellite image.")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY') or "AIzaSyD-mkg__8KYzemqcpe1t-nilBMrVELs1Mc"
    if not api_key:
        print("❌ Gemini API key required. Set GEMINI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Validate API key format
    if not api_key.startswith('AIza'):
        print("⚠️  Warning: API key format doesn't look like a valid Gemini API key")
        print("   Expected format: AIzaSy...")
        print("   Continuing anyway...")
    
    print(f"🔑 Using Gemini API key: {api_key[:10]}...{api_key[-4:]}")
    
    # Initialize detector
    detector = SatelliteChangeDetector(api_key)
    
    # Process images
    success = detector.process_images(args.image1, args.image2, args.output)
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print(f"Check the '{args.output}' directory for results.")
    else:
        print("\n❌ Analysis failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main() 