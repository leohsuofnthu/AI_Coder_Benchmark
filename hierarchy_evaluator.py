"""
Hierarchy Evaluation Module
Extracts hierarchical codebook structures from XML for visual comparison.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional


class HierarchyNode:
    """Represents a node in the hierarchy tree."""
    def __init__(self, key: str, description: str, depth: int, is_net: bool, ordinal: int):
        self.key = key
        self.description = description
        self.depth = depth
        self.is_net = is_net
        self.ordinal = ordinal
        self.children: List['HierarchyNode'] = []
        self.parent: Optional['HierarchyNode'] = None
        
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'key': self.key,
            'description': self.description,
            'depth': self.depth,
            'is_net': self.is_net,
            'children': [child.to_dict() for child in self.children]
        }


class HierarchyEvaluator:
    """Extracts hierarchical codebook structures for visual comparison."""
    
    def __init__(self, benchmark_path: str, model_path: str):
        self.benchmark_path = benchmark_path
        self.model_path = model_path
        self.benchmark_tree: Optional[HierarchyNode] = None
        self.model_tree: Optional[HierarchyNode] = None
        
    def _get_text(self, element, tag: str, default: str = '') -> str:
        """Safely extract text from XML element."""
        child = element.find(tag)
        return child.text if child is not None and child.text else default
    
    def _get_int(self, element, tag: str, default: int = 0) -> int:
        """Safely extract integer from XML element."""
        text = self._get_text(element, tag, str(default))
        try:
            return int(text)
        except ValueError:
            return default
    
    def extract_hierarchy(self, xml_path: str) -> List[HierarchyNode]:
        """Extract hierarchy from XML and build tree structure. Only includes codes used in open-ended questions."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # First, find all codes used in open-ended questions (QuestionType = 0)
        codes_used_in_open_ended = set()
        
        for question in root.findall('.//Question'):
            question_type = self._get_text(question, 'QuestionType', '0')
            
            # Only process open-ended questions (QuestionType = 0)
            if question_type != '0':
                continue
            
            # Collect all codes used in responses to this open-ended question
            for response in question.findall('.//Response'):
                for resp_code in response.findall('.//ResponseCode'):
                    cbc_key = self._get_text(resp_code, 'DCCBCKey')
                    if cbc_key:
                        codes_used_in_open_ended.add(cbc_key)
        
        # Now collect only codes that are used in open-ended questions
        codes = []
        code_dict = {}  # Store all codes first, then filter by usage
        
        for codebook in root.findall('.//CodeBook'):
            for code in codebook.findall('.//CodeBookCode'):
                key = self._get_text(code, 'CBCKey')
                if not key:
                    continue
                
                description = self._get_text(code, 'CBCDescription')
                
                # Filter out "Uncoded Segment" codes (same as main evaluator)
                desc_norm = (description or '').strip().lower()
                if "uncoded" in desc_norm and "segment" in desc_norm:
                    continue  # Skip this code - it's a placeholder
                
                depth = self._get_int(code, 'CBCDepth', 1)
                is_net = self._get_text(code, 'CBCIsNet') == 'True'
                ordinal = self._get_int(code, 'CBCOrdinal', 0)
                
                code_node = HierarchyNode(key, description, depth, is_net, ordinal)
                code_dict[key] = code_node
        
        # Build full tree first to establish parent-child relationships
        all_codes = list(code_dict.values())
        all_codes.sort(key=lambda x: x.ordinal)
        
        # Build temporary tree structure to find ancestors
        stack = []  # Stack of parents at each depth level
        
        for code in all_codes:
            # Clear stack entries at same or deeper depth
            while stack and stack[-1].depth >= code.depth:
                stack.pop()
            
            if code.depth == 1:
                stack = [code]
            elif stack:
                code.parent = stack[-1]
                stack.append(code)
            else:
                stack = [code]
        
        # Now find all codes to include: used codes + all their ancestors
        codes_to_include = set(codes_used_in_open_ended)
        
        # For each used code, traverse up the hierarchy to include all ancestors
        for code_key in codes_used_in_open_ended:
            if code_key not in code_dict:
                continue
            current = code_dict[code_key]
            while current:
                codes_to_include.add(current.key)
                # Move to parent
                if hasattr(current, 'parent') and current.parent:
                    current = current.parent
                else:
                    # Find parent by depth (most recent code at depth-1 before this code)
                    parent_found = False
                    for other_code in all_codes:
                        if (other_code.ordinal < current.ordinal and 
                            other_code.depth == current.depth - 1):
                            codes_to_include.add(other_code.key)
                            current = other_code
                            parent_found = True
                            break
                    if not parent_found:
                        break
        
        # Filter codes to only those used in open-ended questions or their ancestors
        codes = [code_dict[key] for key in codes_to_include if key in code_dict]
        
        # If no codes found, fall back to all codes (in case no open-ended questions found)
        if not codes:
            codes = all_codes
        
        # Sort by ordinal to maintain order
        codes.sort(key=lambda x: x.ordinal)
        
        # Build tree: child at depth N belongs to most recent parent at depth N-1
        root_nodes = []
        stack = []  # Stack of parents at each depth level
        
        for code in codes:
            # Clear stack entries at same or deeper depth
            while stack and stack[-1].depth >= code.depth:
                stack.pop()
            
            if code.depth == 1:
                # Root level node
                root_nodes.append(code)
                stack = [code]
            elif stack:
                # Add as child of most recent parent
                parent = stack[-1]
                parent.children.append(code)
                code.parent = parent
                stack.append(code)
            else:
                # Orphan node - add as root
                root_nodes.append(code)
                stack = [code]
        
        return root_nodes
    
    def evaluate(self) -> Dict:
        """Extract hierarchy structures for visual comparison."""
        # Extract hierarchies
        benchmark_roots = self.extract_hierarchy(self.benchmark_path)
        model_roots = self.extract_hierarchy(self.model_path)
        
        # Convert to single root if multiple roots
        if len(benchmark_roots) == 1:
            self.benchmark_tree = benchmark_roots[0]
        else:
            # Create virtual root
            virtual_root = HierarchyNode('root', 'Root', 0, False, 0)
            virtual_root.children = benchmark_roots
            self.benchmark_tree = virtual_root
        
        if len(model_roots) == 1:
            self.model_tree = model_roots[0]
        else:
            virtual_root = HierarchyNode('root', 'Root', 0, False, 0)
            virtual_root.children = model_roots
            self.model_tree = virtual_root
        
        return {
            'benchmark': {
                'tree': self.benchmark_tree.to_dict() if self.benchmark_tree else {}
            },
            'model': {
                'tree': self.model_tree.to_dict() if self.model_tree else {}
            }
        }
