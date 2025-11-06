"""
Hierarchy Evaluation Module
Extracts hierarchical topic structures from XML and calculates comparison metrics.
Uses Sentence-BERT for semantic similarity calculations.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import statistics

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")


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
    """Evaluates hierarchical topic structures."""
    
    def __init__(self, benchmark_path: str, model_path: str):
        self.benchmark_path = benchmark_path
        self.model_path = model_path
        self.benchmark_tree: Optional[HierarchyNode] = None
        self.model_tree: Optional[HierarchyNode] = None
        
        # Load Sentence-BERT model for semantic similarity
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use a lightweight, fast model for semantic similarity
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded Sentence-BERT model: all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Warning: Could not load Sentence-BERT model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
        
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
        """Extract hierarchy from XML and build tree structure."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Collect all codes
        codes = []
        for codebook in root.findall('.//CodeBook'):
            for code in codebook.findall('.//CodeBookCode'):
                key = self._get_text(code, 'CBCKey')
                if not key:
                    continue
                
                description = self._get_text(code, 'CBCDescription')
                depth = self._get_int(code, 'CBCDepth', 1)
                is_net = self._get_text(code, 'CBCIsNet') == 'True'
                ordinal = self._get_int(code, 'CBCOrdinal', 0)
                
                codes.append(HierarchyNode(key, description, depth, is_net, ordinal))
        
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
    
    def _build_flat_dict(self, nodes: List[HierarchyNode]) -> Dict[str, HierarchyNode]:
        """Build flat dictionary of all nodes."""
        flat = {}
        for node in nodes:
            flat[node.key] = node
            for child in node.children:
                flat.update(self._build_flat_dict([child]))
        return flat
    
    def _calculate_coherence(self, nodes: List[HierarchyNode]) -> float:
        """
        Calculate Coherence: mean cosine(v_i, v_centroid) for each cluster.
        For hierarchy: each parent node's children form a cluster.
        """
        if not self.embedding_model:
            return 0.0
        
        try:
            flat = self._build_flat_dict(nodes)
            all_coherences = []
            
            # For each parent node, calculate coherence of its children
            for node in flat.values():
                if not node.children or len(node.children) < 2:
                    continue
                
                # Get descriptions of children
                child_descriptions = [c.description for c in node.children if c.description]
                if len(child_descriptions) < 2:
                    continue
                
                # Compute embeddings
                embeddings = self.embedding_model.encode(child_descriptions, convert_to_numpy=True, show_progress_bar=False)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                # Calculate centroid
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                
                # Calculate mean cosine similarity to centroid
                cosines = np.dot(embeddings, centroid)
                mean_coherence = float(np.mean(cosines))
                all_coherences.append(mean_coherence)
            
            return round(statistics.mean(all_coherences), 3) if all_coherences else 0.0
        except Exception as e:
            print(f"Warning: Error calculating coherence: {e}")
            return 0.0
    
    def _calculate_overlap(self, nodes: List[HierarchyNode]) -> float:
        """
        Calculate Overlap: |S_i ∩ S_j| / |S_i ∪ S_j| for sibling clusters.
        Since we don't have assignment data, we approximate using node descriptions.
        """
        if not self.embedding_model:
            return 0.0
        
        try:
            flat = self._build_flat_dict(nodes)
            all_overlaps = []
            
            # For each parent, compare sibling children
            for node in flat.values():
                siblings = node.children
                if len(siblings) < 2:
                    continue
                
                # Get sibling descriptions
                sibling_descs = [s.description for s in siblings if s.description]
                if len(sibling_descs) < 2:
                    continue
                
                # Compute embeddings
                embeddings = self.embedding_model.encode(sibling_descs, convert_to_numpy=True, show_progress_bar=False)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                # Calculate pairwise cosine similarities (approximation of overlap)
                # High similarity = high overlap
                pairwise_sims = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = float(np.dot(embeddings[i], embeddings[j]))
                        pairwise_sims.append(sim)
                
                if pairwise_sims:
                    # Average overlap (high similarity = high overlap)
                    avg_overlap = statistics.mean(pairwise_sims)
                    all_overlaps.append(avg_overlap)
            
            return round(statistics.mean(all_overlaps), 3) if all_overlaps else 0.0
        except Exception as e:
            print(f"Warning: Error calculating overlap: {e}")
            return 0.0
    
    def _calculate_granularity(self, nodes: List[HierarchyNode]) -> Dict[int, float]:
        """
        Calculate Granularity Score G(L) = 1 - mean cosine(v_i, v_centroid_L) for each level.
        Returns dict: {level: granularity_score}
        """
        if not self.embedding_model:
            return {}
        
        try:
            flat = self._build_flat_dict(nodes)
            all_nodes = list(flat.values())
            
            # Group nodes by depth/level
            nodes_by_level = defaultdict(list)
            for node in all_nodes:
                nodes_by_level[node.depth].append(node)
            
            granularity_by_level = {}
            
            for level, level_nodes in nodes_by_level.items():
                if len(level_nodes) < 2:
                    continue
                
                descriptions = [n.description for n in level_nodes if n.description]
                if len(descriptions) < 2:
                    continue
                
                # Compute embeddings
                embeddings = self.embedding_model.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                # Calculate centroid for this level
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                
                # Calculate mean cosine to centroid
                cosines = np.dot(embeddings, centroid)
                mean_cosine = float(np.mean(cosines))
                
                # Granularity = 1 - mean_cosine
                granularity = 1 - mean_cosine
                granularity_by_level[level] = round(granularity, 3)
            
            return granularity_by_level
        except Exception as e:
            print(f"Warning: Error calculating granularity: {e}")
            return {}
    
    def _calculate_consistency(self, nodes: List[HierarchyNode]) -> float:
        """
        Calculate Consistency: mean Parent-Child Distance = 1 - cosine(v_p, v_c).
        Target range: 0.15-0.35
        """
        if not self.embedding_model:
            return 0.0
        
        try:
            flat = self._build_flat_dict(nodes)
            parent_child_distances = []
            
            for node in flat.values():
                if not node.children:
                    continue
                
                # Get parent description
                parent_desc = node.description
                if not parent_desc:
                    continue
                
                # Get child descriptions
                child_descs = [c.description for c in node.children if c.description]
                if not child_descs:
                    continue
                
                # Compute embeddings
                parent_emb = self.embedding_model.encode([parent_desc], convert_to_numpy=True, show_progress_bar=False)[0]
                child_embs = self.embedding_model.encode(child_descs, convert_to_numpy=True, show_progress_bar=False)
                
                # Normalize
                parent_emb = parent_emb / np.linalg.norm(parent_emb)
                child_embs = child_embs / np.linalg.norm(child_embs, axis=1, keepdims=True)
                
                # Calculate distances: 1 - cosine(v_p, v_c)
                for child_emb in child_embs:
                    cosine = float(np.dot(parent_emb, child_emb))
                    distance = 1 - cosine
                    parent_child_distances.append(distance)
            
            return round(statistics.mean(parent_child_distances), 3) if parent_child_distances else 0.0
        except Exception as e:
            print(f"Warning: Error calculating consistency: {e}")
            return 0.0
    
    def _calculate_tree_stats(self, nodes: List[HierarchyNode]) -> Dict:
        """Calculate internal metrics for a hierarchy."""
        if not nodes:
            return {}
        
        flat = self._build_flat_dict(nodes)
        all_nodes = list(flat.values())
        
        # Depth statistics
        depths = [n.depth for n in all_nodes]
        max_depth = max(depths) if depths else 0
        depth_counts = defaultdict(int)
        for d in depths:
            depth_counts[d] += 1
        
        # Branching factor (average children per parent)
        parents = [n for n in all_nodes if n.children]
        branching_factors = [len(p.children) for p in parents]
        avg_branching = statistics.mean(branching_factors) if branching_factors else 0
        
        # Depth variance (balance)
        branch_depths = []
        def get_branch_depth(node, current=0):
            if not node.children:
                branch_depths.append(current)
            else:
                for child in node.children:
                    get_branch_depth(child, current + 1)
        
        for root in nodes:
            get_branch_depth(root)
        
        if len(branch_depths) > 1:
            try:
                depth_variance = statistics.stdev(branch_depths)
            except:
                depth_variance = 0
        else:
            depth_variance = 0
        
        # Coverage: count leaf nodes (codes that can be assigned)
        leaf_nodes = [n for n in all_nodes if not n.children]
        total_nodes = len(all_nodes)
        
        # Calculate semantic metrics (if embeddings available)
        coherence = self._calculate_coherence(nodes)
        overlap = self._calculate_overlap(nodes)
        granularity = self._calculate_granularity(nodes)
        consistency = self._calculate_consistency(nodes)
        
        stats = {
            'total_nodes': total_nodes,
            'leaf_nodes': len(leaf_nodes),
            'max_depth': max_depth,
            'depth_distribution': dict(depth_counts),
            'avg_branching_factor': round(avg_branching, 2),
            'depth_variance': round(depth_variance, 2),
            'net_nodes': len([n for n in all_nodes if n.is_net]),
            'code_nodes': len([n for n in all_nodes if not n.is_net])
        }
        
        # Add semantic metrics if available
        if coherence > 0:
            stats['coherence'] = coherence
        if overlap > 0:
            stats['overlap'] = overlap
        if granularity:
            stats['granularity'] = granularity
        if consistency > 0:
            stats['consistency'] = consistency
        
        return stats
    
    def _calculate_tree_edit_distance(self, tree1: List[HierarchyNode], tree2: List[HierarchyNode]) -> int:
        """Calculate approximate tree edit distance (number of structural differences)."""
        flat1 = self._build_flat_dict(tree1)
        flat2 = self._build_flat_dict(tree2)
        
        keys1 = set(flat1.keys())
        keys2 = set(flat2.keys())
        
        # Missing/extra nodes
        missing = len(keys1 - keys2)
        extra = len(keys2 - keys1)
        
        # Structural differences (parent-child mismatches)
        structural_diff = 0
        common_keys = keys1 & keys2
        for key in common_keys:
            node1 = flat1[key]
            node2 = flat2[key]
            
            # Check if parents match
            parent1 = node1.parent.key if node1.parent else None
            parent2 = node2.parent.key if node2.parent else None
            if parent1 != parent2:
                structural_diff += 1
            
            # Check if children match
            children1 = {c.key for c in node1.children}
            children2 = {c.key for c in node2.children}
            if children1 != children2:
                structural_diff += 1
        
        return missing + extra + structural_diff
    
    def _calculate_node_similarity(self, tree1: List[HierarchyNode], tree2: List[HierarchyNode]) -> float:
        """
        Calculate node-level semantic similarity using Sentence-BERT embeddings.
        Returns mean of max cosine similarity (model → benchmark) as per spec.
        """
        flat1 = self._build_flat_dict(tree1)
        flat2 = self._build_flat_dict(tree2)
        
        keys1 = set(flat1.keys())
        keys2 = set(flat2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        # If Sentence-BERT is available, use embeddings
        if self.embedding_model:
            try:
                # Get descriptions for all nodes
                descriptions1 = {k: flat1[k].description for k in keys1 if flat1[k].description}
                descriptions2 = {k: flat2[k].description for k in keys2 if flat2[k].description}
                
                if not descriptions1 or not descriptions2:
                    return 0.0
                
                # Compute embeddings
                texts1 = list(descriptions1.values())
                texts2 = list(descriptions2.values())
                
                embeddings1 = self.embedding_model.encode(texts1, convert_to_numpy=True, show_progress_bar=False)
                embeddings2 = self.embedding_model.encode(texts2, convert_to_numpy=True, show_progress_bar=False)
                
                # Normalize embeddings for cosine similarity
                embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
                embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
                
                # Calculate similarity: for each model node, find max similarity to benchmark nodes
                # This implements "Mean of max cosine(model → human)" from spec
                max_similarities = []
                
                # Create mapping: key -> index
                idx_to_key1 = {i: k for i, k in enumerate(descriptions1.keys())}
                idx_to_key2 = {i: k for i, k in enumerate(descriptions2.keys())}
                
                # For each model node (tree2), find max similarity to benchmark nodes (tree1)
                for i in range(len(embeddings2)):
                    # Cosine similarity with all benchmark embeddings
                    similarities = np.dot(embeddings2[i], embeddings1.T)
                    max_sim = float(np.max(similarities))
                    max_similarities.append(max_sim)
                
                # Mean of max similarities
                mean_similarity = float(np.mean(max_similarities)) if max_similarities else 0.0
                
                # Also add coverage component (exact key matches)
                common_keys = keys1 & keys2
                coverage = len(common_keys) / max(len(keys1), len(keys2)) if keys1 or keys2 else 0
                
                # Weighted combination: 80% semantic similarity, 20% coverage
                return round(mean_similarity * 0.8 + coverage * 0.2, 3)
                
            except Exception as e:
                print(f"Warning: Error computing embeddings, falling back to word overlap: {e}")
                # Fall through to word overlap
        
        # Fallback: word overlap (simple Jaccard on words)
        similarity_scores = []
        for key in keys1 & keys2:
            desc1 = flat1[key].description.lower()
            desc2 = flat2[key].description.lower()
            
            words1 = set(desc1.split())
            words2 = set(desc2.split())
            
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1 | words2)
                similarity_scores.append(overlap)
        
        avg_similarity = statistics.mean(similarity_scores) if similarity_scores else 0
        coverage = len(keys1 & keys2) / max(len(keys1), len(keys2)) if keys1 or keys2 else 0
        
        return round((avg_similarity * 0.7 + coverage * 0.3), 3)
    
    def _calculate_ancestor_descendant_agreement(self, tree1: List[HierarchyNode], tree2: List[HierarchyNode]) -> float:
        """Calculate ancestor-descendant relationship agreement."""
        flat1 = self._build_flat_dict(tree1)
        flat2 = self._build_flat_dict(tree2)
        
        def get_ancestors(node):
            ancestors = []
            current = node.parent
            while current:
                ancestors.append(current.key)
                current = current.parent
            return set(ancestors)
        
        common_keys = set(flat1.keys()) & set(flat2.keys())
        if not common_keys:
            return 0.0
        
        agreements = 0
        for key in common_keys:
            node1 = flat1[key]
            node2 = flat2[key]
            
            ancestors1 = get_ancestors(node1)
            ancestors2 = get_ancestors(node2)
            
            if ancestors1 == ancestors2:
                agreements += 1
        
        return round(agreements / len(common_keys), 3) if common_keys else 0.0
    
    def _calculate_path_based_jaccard(self, tree1: List[HierarchyNode], tree2: List[HierarchyNode]) -> float:
        """Calculate Path-based Jaccard: Jaccard of root→leaf path sets."""
        def get_paths(nodes, current_path=None):
            """Get all root-to-leaf paths."""
            if current_path is None:
                current_path = []
            
            paths = []
            for node in nodes:
                path = current_path + [node.key]
                
                if not node.children:
                    # Leaf node - add path
                    paths.append(tuple(path))
                else:
                    # Continue down tree
                    paths.extend(get_paths(node.children, path))
            
            return paths
        
        paths1 = set(get_paths(tree1))
        paths2 = set(get_paths(tree2))
        
        if not paths1 and not paths2:
            return 1.0
        
        intersection = len(paths1 & paths2)
        union = len(paths1 | paths2)
        
        return round(intersection / union, 3) if union > 0 else 0.0
    
    def evaluate(self) -> Dict:
        """Run complete hierarchy evaluation (structure-only metrics)."""
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
        
        # Calculate internal metrics
        benchmark_stats = self._calculate_tree_stats(benchmark_roots)
        model_stats = self._calculate_tree_stats(model_roots)
        
        # Calculate comparison metrics
        tree_edit_distance = self._calculate_tree_edit_distance(benchmark_roots, model_roots)
        node_similarity = self._calculate_node_similarity(benchmark_roots, model_roots)
        ancestor_agreement = self._calculate_ancestor_descendant_agreement(benchmark_roots, model_roots)
        path_jaccard = self._calculate_path_based_jaccard(benchmark_roots, model_roots)
        
        # Delta metrics (structure only)
        delta_depth_variance = model_stats.get('depth_variance', 0) - benchmark_stats.get('depth_variance', 0)
        delta_branching = model_stats.get('avg_branching_factor', 0) - benchmark_stats.get('avg_branching_factor', 0)
        
        # Delta metrics from spec (lines 49-52)
        benchmark_coherence = benchmark_stats.get('coherence', 0)
        model_coherence = model_stats.get('coherence', 0)
        delta_coherence = model_coherence - benchmark_coherence if benchmark_coherence > 0 or model_coherence > 0 else None
        
        benchmark_overlap = benchmark_stats.get('overlap', 0)
        model_overlap = model_stats.get('overlap', 0)
        delta_overlap = benchmark_overlap - model_overlap if benchmark_overlap > 0 or model_overlap > 0 else None
        
        benchmark_granularity = benchmark_stats.get('granularity', {})
        model_granularity = model_stats.get('granularity', {})
        # Average granularity scores for comparison
        avg_benchmark_granularity = statistics.mean(benchmark_granularity.values()) if benchmark_granularity else None
        avg_model_granularity = statistics.mean(model_granularity.values()) if model_granularity else None
        delta_granularity = (avg_model_granularity - avg_benchmark_granularity) if (avg_benchmark_granularity is not None and avg_model_granularity is not None) else None
        
        benchmark_consistency = benchmark_stats.get('consistency', 0)
        model_consistency = model_stats.get('consistency', 0)
        delta_consistency = model_consistency - benchmark_consistency if benchmark_consistency > 0 or model_consistency > 0 else None
        
        # HQI (Hierarchy Quality Index) - per spec formula (line 53)
        # Formula: 0.3 S_node + 0.2 (1–TED) + 0.2 C + 0.2 Coherence + 0.1 Stability
        # Since we removed C (Coverage) and Stability needs multiple runs, we approximate:
        ted_normalized = 1 - min(tree_edit_distance / max(benchmark_stats.get('total_nodes', 1), 1), 1.0)
        
        # Use benchmark coherence (or fallback to 0.75 if not available)
        coherence_value = benchmark_coherence if benchmark_coherence > 0 else 0.75
        
        # Stability approximation: use ancestor agreement as proxy for structural stability
        stability_proxy = ancestor_agreement
        
        # Updated HQI formula (as close to spec as possible without C)
        hqi = round(
            0.3 * node_similarity +  # S_node (Node Semantic Similarity)
            0.2 * ted_normalized +  # (1 - TED)
            0.2 * coherence_value +  # Coherence (using benchmark)
            0.2 * ancestor_agreement +  # Approximation: ADA as structure quality
            0.1 * stability_proxy,  # Stability proxy
            3
        )
        
        return {
            'benchmark': {
                'tree': self.benchmark_tree.to_dict() if self.benchmark_tree else {},
                'stats': benchmark_stats
            },
            'model': {
                'tree': self.model_tree.to_dict() if self.model_tree else {},
                'stats': model_stats
            },
            'comparison': {
                'tree_edit_distance': tree_edit_distance,
                'node_similarity': node_similarity,
                'ancestor_descendant_agreement': ancestor_agreement,
                'path_based_jaccard': path_jaccard,
                'delta_depth_variance': round(delta_depth_variance, 2),
                'delta_branching_factor': round(delta_branching, 2),
                'delta_coherence': round(delta_coherence, 3) if delta_coherence is not None else None,
                'delta_overlap': round(delta_overlap, 3) if delta_overlap is not None else None,
                'delta_granularity': round(delta_granularity, 3) if delta_granularity is not None else None,
                'delta_consistency': round(delta_consistency, 3) if delta_consistency is not None else None,
                'hqi': hqi
            }
        }

