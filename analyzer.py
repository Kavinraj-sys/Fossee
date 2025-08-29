# Student Python Competence Analysis System
# Using Open Source Models for Educational Assessment

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ast
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class CompetenceAnalysis:
    """Structure for storing competence analysis results"""
    concept_understanding: float
    identified_gaps: List[str]
    generated_prompts: List[str]
    misconceptions: List[str]
    suggestions: List[str]

class StudentCompetenceAnalyzer:
    """
    Main class for analyzing student Python code and generating
    competence assessments using open source models
    """
    
    def __init__(self, model_name: str = "Salesforce/codet5p-220m"):
        """
        Initialize the analyzer with specified model
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.concept_map = self._load_concept_map()
        
    def _load_concept_map(self) -> Dict:
        """Load Python concept hierarchy for assessment"""
        return {
            "basic": ["variables", "data_types", "operators", "control_flow"],
            "intermediate": ["functions", "lists", "dictionaries", "loops"],
            "advanced": ["classes", "inheritance", "decorators", "generators"],
            "expert": ["metaclasses", "context_managers", "async", "optimization"]
        }
    
    def analyze_code(self, student_code: str) -> CompetenceAnalysis:
        """
        Analyze student Python code for competence indicators
        
        Args:
            student_code: Python code submitted by student
            
        Returns:
            CompetenceAnalysis object with assessment results
        """
        # Parse code structure
        code_features = self._extract_code_features(student_code)
        
        # Generate code understanding
        understanding_score = self._assess_understanding(student_code, code_features)
        
        # Identify conceptual gaps
        gaps = self._identify_gaps(code_features)
        
        # Generate educational prompts
        prompts = self._generate_prompts(student_code, gaps)
        
        # Detect misconceptions
        misconceptions = self._detect_misconceptions(student_code)
        
        # Create improvement suggestions
        suggestions = self._generate_suggestions(gaps, misconceptions)
        
        return CompetenceAnalysis(
            concept_understanding=understanding_score,
            identified_gaps=gaps,
            generated_prompts=prompts,
            misconceptions=misconceptions,
            suggestions=suggestions
        )
    
    def _extract_code_features(self, code: str) -> Dict:
        """Extract structural and semantic features from code"""
        features = {
            "complexity": 0,
            "concepts_used": [],
            "patterns": [],
            "potential_issues": []
        }
        
        try:
            tree = ast.parse(code)
            
            # Analyze AST for features
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    features["concepts_used"].append("functions")
                    features["complexity"] += 1
                elif isinstance(node, ast.ClassDef):
                    features["concepts_used"].append("classes")
                    features["complexity"] += 2
                elif isinstance(node, ast.For):
                    features["concepts_used"].append("loops")
                elif isinstance(node, ast.If):
                    features["concepts_used"].append("conditionals")
                elif isinstance(node, ast.Lambda):
                    features["concepts_used"].append("lambda")
                    features["complexity"] += 1
                    
            # Check for common patterns
            if "list comprehension" in code:
                features["patterns"].append("list_comprehension")
            if "with" in code:
                features["patterns"].append("context_manager")
                
        except SyntaxError as e:
            features["potential_issues"].append(f"Syntax error: {e}")
            
        return features
    
    def _assess_understanding(self, code: str, features: Dict) -> float:
        """
        Assess overall understanding level based on code analysis
        
        Returns:
            Score between 0 and 1 indicating understanding level
        """
        # Use model to analyze code quality
        inputs = self.tokenizer(
            f"Analyze Python code quality: {code[:500]}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=50)
            analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate score based on features and model output
        base_score = min(len(features["concepts_used"]) / 10, 0.5)
        complexity_bonus = min(features["complexity"] / 20, 0.3)
        pattern_bonus = len(features["patterns"]) * 0.05
        issue_penalty = len(features["potential_issues"]) * 0.1
        
        score = base_score + complexity_bonus + pattern_bonus - issue_penalty
        return max(0, min(1, score))
    
    def _identify_gaps(self, features: Dict) -> List[str]:
        """Identify conceptual gaps in student's knowledge"""
        gaps = []
        
        # Check for missing fundamental concepts
        if "functions" not in features["concepts_used"]:
            gaps.append("Function definition and usage")
        if "error_handling" not in features["concepts_used"]:
            gaps.append("Exception handling")
        if not features["patterns"]:
            gaps.append("Python idioms and patterns")
            
        # Check for optimization opportunities
        if features["complexity"] > 10 and "list_comprehension" not in features["patterns"]:
            gaps.append("Code optimization techniques")
            
        return gaps
    
    def _generate_prompts(self, code: str, gaps: List[str]) -> List[str]:
        """Generate educational prompts based on identified gaps"""
        prompts = []
        
        for gap in gaps:
            if gap == "Function definition and usage":
                prompts.append(
                    "How could you refactor this code using functions to improve "
                    "reusability? Consider which parts of your code perform "
                    "specific tasks that could be isolated."
                )
            elif gap == "Exception handling":
                prompts.append(
                    "What might go wrong when this code runs with unexpected input? "
                    "How would you make it more robust?"
                )
            elif gap == "Python idioms and patterns":
                prompts.append(
                    "Research Python's 'pythonic' way of solving this problem. "
                    "What built-in functions or patterns could simplify your approach?"
                )
            elif gap == "Code optimization techniques":
                prompts.append(
                    "Analyze the time complexity of your solution. Could you achieve "
                    "the same result with fewer iterations or more efficient data structures?"
                )
                
        # Generate model-based prompts
        model_prompt = self._generate_model_prompt(code)
        if model_prompt:
            prompts.append(model_prompt)
            
        return prompts
    
    def _generate_model_prompt(self, code: str) -> str:
        """Use model to generate a contextual learning prompt"""
        inputs = self.tokenizer(
            f"Generate educational question about this Python code without revealing solution: {code[:300]}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
            prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        return prompt if len(prompt) > 20 else ""
    
    def _detect_misconceptions(self, code: str) -> List[str]:
        """Detect common misconceptions in student code"""
        misconceptions = []
        
        # Check for common Python misconceptions
        if "range(len(" in code:
            misconceptions.append(
                "Using range(len()) instead of direct iteration - "
                "Consider iterating directly over the sequence"
            )
        if "== True" in code or "== False" in code:
            misconceptions.append(
                "Explicitly comparing to boolean values - "
                "Boolean expressions can be used directly"
            )
        if "except:" in code and "except Exception" not in code:
            misconceptions.append(
                "Using bare except clause - "
                "Specify exception types for better error handling"
            )
            
        return misconceptions
    
    def _generate_suggestions(self, gaps: List[str], misconceptions: List[str]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if gaps:
            suggestions.append(
                f"Focus on learning: {', '.join(gaps[:2])} to strengthen your foundation"
            )
        if misconceptions:
            suggestions.append(
                "Review Python best practices and idioms to write more pythonic code"
            )
        
        suggestions.append(
            "Practice code review - reading others' solutions can provide new perspectives"
        )
        
        return suggestions

# Example usage and testing
def test_analyzer():
    """Test the competence analyzer with sample student code"""
    
    # Sample student code (basic level)
    student_code_basic = """
def calculate_sum(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total

result = calculate_sum([1, 2, 3, 4, 5])
print(result)
    """
    
    # Sample student code (intermediate level)
    student_code_intermediate = """
class Student:
    def __init__(self, name, grades):
        self.name = name
        self.grades = grades
    
    def average_grade(self):
        return sum(self.grades) / len(self.grades)
    
students = [Student("Alice", [90, 85, 88]), Student("Bob", [75, 80, 82])]
for student in students:
    print(f"{student.name}: {student.average_grade()}")
    """
    
    # Initialize analyzer
    analyzer = StudentCompetenceAnalyzer()
    
    # Analyze basic code
    print("=== Basic Code Analysis ===")
    analysis_basic = analyzer.analyze_code(student_code_basic)
    print(f"Understanding Score: {analysis_basic.concept_understanding:.2f}")
    print(f"Identified Gaps: {analysis_basic.identified_gaps}")
    print(f"Generated Prompts: {analysis_basic.generated_prompts}")
    print(f"Misconceptions: {analysis_basic.misconceptions}")
    print()
    
    # Analyze intermediate code
    print("=== Intermediate Code Analysis ===")
    analysis_intermediate = analyzer.analyze_code(student_code_intermediate)
    print(f"Understanding Score: {analysis_intermediate.concept_understanding:.2f}")
    print(f"Identified Gaps: {analysis_intermediate.identified_gaps}")
    print(f"Generated Prompts: {analysis_intermediate.generated_prompts}")
    
    return analysis_basic, analysis_intermediate

if __name__ == "__main__":
    # Run the test
    test_analyzer()
