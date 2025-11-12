"""
Hierarchy Evaluation Module
Extracts hierarchical codebook structures from XML for visual comparison.
Computes semantic metrics using sentence embeddings.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


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


class HierarchyMetricsEvaluator:
    """
    Computes semantic metrics for hierarchical codebook structures.
    Uses sentence embeddings to evaluate coherence, similarity, diversity, and granularity.
    """
    
    # Model selection: Using all-MiniLM-L6-v2 for CPU efficiency
    # Alternative: "intfloat/e5-base-v2" (better quality but slower, requires more memory)
    # Note: all-MiniLM-L6-v2 is ~90MB and 2-5x faster than e5-base on CPU
    # For production with GPU, consider switching to e5-base-v2 for better embeddings
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(self, progress_callback=None):
        """
        Initialize the sentence transformer model (lazy loading).
        
        Args:
            progress_callback: Deprecated - no longer used (kept for backward compatibility)
        """
        self.model = None
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if not self._model_loaded:
            try:
                print("[METRICS] Loading sentence transformer model...")
                self.model = SentenceTransformer(self.MODEL_NAME)
                self._model_loaded = True
                print(f"[METRICS] Model loaded: {self.MODEL_NAME}")
            except Exception as e:
                print(f"[METRICS] Error loading model: {e}")
                raise
    
    def _embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of texts using the sentence transformer model.
        Returns float32 arrays to save 50% memory compared to float64.
        """
        if not texts:
            return np.array([], dtype=np.float32)
        self._load_model()
        # Filter empty texts
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            return np.array([], dtype=np.float32)
        embeddings = self.model.encode(
            non_empty_texts, 
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        # Convert to float32 to save 50% memory (384 bytes -> 192 bytes per embedding)
        # Precision loss is negligible for similarity calculations
        return embeddings.astype(np.float32)
    
    def _get_all_nodes(self, root: HierarchyNode) -> List[HierarchyNode]:
        """Recursively collect all nodes from the tree."""
        nodes = [root]
        for child in root.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def _get_nodes_by_level(self, root: HierarchyNode) -> Dict[int, List[HierarchyNode]]:
        """Group nodes by depth level."""
        levels = defaultdict(list)
        
        def traverse(node, depth):
            levels[depth].append(node)
            for child in node.children:
                traverse(child, depth + 1)
        
        traverse(root, root.depth)
        return dict(levels)
    
    def _get_text_safe(self, element, tag: str, default: str = '') -> str:
        """Safely extract text from XML element."""
        child = element.find(tag)
        return child.text if child is not None and child.text else default
    
    def _extract_documents(self, xml_path: str) -> Dict[str, List[str]]:
        """Extract documents per code from XML file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        code_to_documents = defaultdict(list)
        codebook = {}
        
        # Get codebook
        for codebook_elem in root.findall('.//CodeBook'):
            for code in codebook_elem.findall('.//CodeBookCode'):
                key = self._get_text_safe(code, 'CBCKey')
                if key:
                    description = self._get_text_safe(code, 'CBCDescription')
                    desc_norm = (description or '').strip().lower()
                    if not ("uncoded" in desc_norm and "segment" in desc_norm):
                        codebook[key] = description
        
        # Extract responses and map to codes
        for question in root.findall('.//Question'):
            question_type = self._get_text_safe(question, 'QuestionType', '0')
            if question_type != '0':
                continue
            
            for response in question.findall('.//Response'):
                verbatim = self._get_text_safe(response, 'DROVerbatim', '').strip()
                if not verbatim:
                    continue
                
                codes = set()
                for resp_code in response.findall('.//ResponseCode'):
                    cbc_key = self._get_text_safe(resp_code, 'DCCBCKey')
                    if cbc_key and cbc_key in codebook:
                        codes.add(cbc_key)
                
                for code_key in codes:
                    code_to_documents[code_key].append(verbatim)
        
        return dict(code_to_documents)
    
    
    def _generate_metrics_summary(self, metrics: Dict) -> Dict:
        """
        Generate comprehensive metrics summary with expected ranges and interpretations.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            Dictionary with metric summaries including expected ranges and interpretations
        """
        summary = {
            'embedding_coherence': {
                'value': metrics['embedding_coherence'],
                'expected_range': '≥0.75 (good)',
                'interpretation': 'High coherence (>0.75) indicates tight semantic grouping within topics. Low coherence suggests topics may be too broad or contain unrelated content.',
                'status': 'good' if metrics['embedding_coherence'] >= 0.75 else 'fair' if metrics['embedding_coherence'] >= 0.6 else 'poor'
            },
            'parent_child_similarity': {
                'value': metrics['parent_child_similarity'],
                'expected_range': '0.5-0.8 (typical)',
                'interpretation': 'Measures semantic alignment between parent and child topics. Moderate values (0.5-0.8) indicate proper hierarchical refinement. Too high (>0.9) suggests redundancy; too low (<0.3) suggests weak relationships.',
                'status': 'good' if 0.5 <= metrics['parent_child_similarity'] <= 0.8 else 'fair' if 0.3 <= metrics['parent_child_similarity'] <= 0.9 else 'poor'
            },
            'local_duplicate_penalty': {
                'value': metrics['local_duplicate_penalty'],
                'expected_range': '≤0.10 (good)',
                'interpretation': 'Fraction of sibling pairs with high similarity (>0.85). Low values (<0.1) indicate good separation between sibling topics. High values indicate redundancy among siblings under the same parent.',
                'status': 'good' if metrics['local_duplicate_penalty'] <= 0.10 else 'fair' if metrics['local_duplicate_penalty'] <= 0.20 else 'poor'
            },
            'global_duplicate_penalty': {
                'value': metrics['global_duplicate_penalty'],
                'expected_range': '≤0.15 (good)',
                'interpretation': 'Fraction of topic pairs at the same level with high similarity (>0.85). Low values (<0.15) indicate good level-wide diversity. Higher values suggest redundancy across the entire hierarchy level.',
                'status': 'good' if metrics['global_duplicate_penalty'] <= 0.15 else 'fair' if metrics['global_duplicate_penalty'] <= 0.25 else 'poor'
            },
            'intra_level_diversity': {
                'value': metrics['intra_level_diversity'],
                'expected_range': '0.25-0.6 (good)',
                'interpretation': 'Measures variety across topics at the same hierarchy level. Higher values (0.25-0.6) indicate broader coverage and better separation. Lower values suggest topics are too similar.',
                'status': 'good' if 0.25 <= metrics['intra_level_diversity'] <= 0.6 else 'fair' if 0.15 <= metrics['intra_level_diversity'] <= 0.7 else 'poor'
            },
            'inter_level_granularity': {
                'value': metrics['inter_level_granularity'],
                'expected_range': '0.1-0.3 (ideal)',
                'interpretation': 'Difference between parent-child and sibling similarities. Positive values (0.1-0.3) indicate that deeper levels add meaningful refinement. Negative values suggest siblings are more similar than parent-child, indicating weak hierarchical structure.',
                'status': 'good' if 0.1 <= metrics['inter_level_granularity'] <= 0.3 else 'fair' if 0.0 <= metrics['inter_level_granularity'] <= 0.4 else 'poor'
            },
            'net_purity': {
                'value': metrics['net_purity'],
                'expected_range': '0.6-0.9 (good)',
                'interpretation': 'Content alignment: Semantic cohesion between parent topic labels and child document content. Higher values (0.6-0.9) indicate child documents align well with parent topic semantics. Measures how well actual content fits the parent topic description.',
                'status': 'good' if 0.6 <= metrics['net_purity'] <= 0.9 else 'fair' if 0.4 <= metrics['net_purity'] <= 0.95 else 'poor'
            },
            'net_purity_label': {
                'value': metrics.get('net_purity_label', metrics.get('parent_child_similarity', 0)),
                'expected_range': '0.5-0.8 (typical)',
                'interpretation': 'Structural alignment: Semantic similarity between parent and child topic labels. Same as Parent-Child Similarity, measuring label-to-label relationships in the hierarchy structure.',
                'status': 'good' if 0.5 <= metrics.get('net_purity_label', 0) <= 0.8 else 'fair' if 0.3 <= metrics.get('net_purity_label', 0) <= 0.9 else 'poor'
            },
            'composite_quality_score': {
                'value': metrics['composite_quality_score'],
                'expected_range': '0.7-1.0 (excellent), 0.5-0.7 (good), <0.5 (needs improvement)',
                'interpretation': 'Overall hierarchical quality combining coherence, diversity, and structural relationships. Weighted combination: 30% coherence + 30% (1-duplication) + 20% diversity + 20% parent-child similarity.',
                'status': 'excellent' if metrics['composite_quality_score'] >= 0.7 else 'good' if metrics['composite_quality_score'] >= 0.5 else 'needs_improvement'
            }
        }
        return summary
    
    def compute_metrics(self, root: HierarchyNode, xml_path: str) -> Optional[Dict]:
        """
        Compute all semantic metrics for a hierarchy.
        
        Returns:
            Dictionary with metrics or None if computation fails
        """
        try:
            # Extract documents per code
            code_to_documents = self._extract_documents(xml_path)
            
            # Get all nodes
            all_nodes = self._get_all_nodes(root)
            if not all_nodes:
                return None
            
            # Filter out virtual root node
            nodes = [n for n in all_nodes if n.key != 'root']
            if not nodes:
                return None
            
            # Get nodes by level
            nodes_by_level = self._get_nodes_by_level(root)
            max_depth = max(nodes_by_level.keys()) if nodes_by_level else 1
            
            # 1. Embedding Coherence (within subtopics)
            self._load_model()  # Pre-load model
            
            coherence_scores = []
            code_embeddings = {}
            document_centroids = {}
            
            # Embed code descriptions (batch for efficiency)
            code_descriptions = [(node.key, node.description or "") for node in nodes if node.description]
            codes_with_embeddings_count = 0
            if code_descriptions:
                # Batch embed all descriptions at once (returns float32)
                descriptions_list = [desc for _, desc in code_descriptions]
                all_code_embs = self._embed_texts(descriptions_list)
                
                # Map embeddings back to codes
                for idx, (code_key, _) in enumerate(code_descriptions):
                    if idx < len(all_code_embs):
                        code_embeddings[code_key] = all_code_embs[idx]
                        codes_with_embeddings_count += 1
                # Clear temporary lists after mapping
                del all_code_embs
                del code_descriptions, descriptions_list
            
            # Embed documents and compute coherence
            codes_with_docs = [(key, docs) for key, docs in code_to_documents.items() if docs]
            codes_with_docs_count = len(codes_with_docs)
            codes_with_both_count = len([key for key, _ in codes_with_docs if key in code_embeddings])
            single_doc_count = 0
            multi_doc_count = 0
            
            for idx, (code_key, documents) in enumerate(codes_with_docs):
                if documents:
                    # Embed documents
                    doc_embeddings = self._embed_texts(documents)
                    if len(doc_embeddings) > 0:
                        # Compute centroid (already float32 from _embed_texts)
                        centroid = np.mean(doc_embeddings, axis=0)
                        document_centroids[code_key] = centroid
                        
                        # Only compute coherence for codes with 2+ documents (single-doc always = 1.0)
                        if len(doc_embeddings) >= 2:
                            # Compute coherence: mean cosine similarity between documents and centroid
                            similarities = cosine_similarity(doc_embeddings, centroid.reshape(1, -1))
                            coherence = float(np.mean(similarities))
                            coherence_scores.append(coherence)
                            multi_doc_count += 1
                            # Clear similarity matrix immediately
                            del similarities
                        else:
                            single_doc_count += 1
                    
                    # CRITICAL: Delete document embeddings immediately after computing centroid
                    # This frees 50-70% of memory used for document processing
                    del doc_embeddings
            
            embedding_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
            
            # 2. Parent-Child Similarity (mean cosine) - using code descriptions
            parent_child_similarities = []
            total_parent_child_pairs = 0
            for node in nodes:
                if node.parent and node.parent.key != 'root':
                    total_parent_child_pairs += 1
                    if node.key in code_embeddings:
                        parent_key = node.parent.key
                        if parent_key in code_embeddings:
                            parent_emb = code_embeddings[parent_key].reshape(1, -1)
                            child_emb = code_embeddings[node.key].reshape(1, -1)
                            sim = cosine_similarity(parent_emb, child_emb)[0][0]
                            parent_child_similarities.append(float(sim))
            
            parent_child_similarity = float(np.mean(parent_child_similarities)) if parent_child_similarities else 0.0
            parent_child_coverage = len(parent_child_similarities) / total_parent_child_pairs if total_parent_child_pairs > 0 else 0.0
            
            # OPTIMIZATION: Compute sibling similarities ONCE and reuse for multiple metrics
            # This avoids recomputing the same similarities 3 times (saves memory and time)
            sibling_similarity_cache = {}  # (parent_key, tuple(sorted_sibling_keys)) -> upper_triangle array
            
            def compute_and_cache_sibling_similarities(parent_key, siblings, code_embeddings):
                """Compute sibling similarities and cache for reuse."""
                sibling_keys = tuple(sorted(s.key for s in siblings))
                cache_key = (parent_key, sibling_keys)
                
                if cache_key in sibling_similarity_cache:
                    return sibling_similarity_cache[cache_key]
                
                sibling_embeddings = []
                for sibling in siblings:
                    if sibling.key in code_embeddings:
                        sibling_embeddings.append(code_embeddings[sibling.key])
                
                if len(sibling_embeddings) < 2:
                    return None
                
                emb_matrix = np.array(sibling_embeddings, dtype=np.float32)
                similarities = cosine_similarity(emb_matrix)
                upper_triangle = similarities[np.triu_indices(len(similarities), k=1)].astype(np.float32)
                
                # Cache the result (float32 to save memory)
                sibling_similarity_cache[cache_key] = upper_triangle
                
                # Clear intermediate arrays
                del emb_matrix, similarities
                
                return upper_triangle
            
            # 3. Local Duplication Penalty (fraction of sibling pairs with cosine > 0.85)
            local_duplicate_count = 0
            local_total_pairs = 0
            local_total_sibling_groups = 0
            local_groups_with_embeddings = 0
            
            for level, level_nodes in nodes_by_level.items():
                if level == 0:
                    continue
                siblings_by_parent = defaultdict(list)
                for node in level_nodes:
                    if node.parent and node.parent.key != 'root':
                        siblings_by_parent[node.parent.key].append(node)
                
                # Compute duplication for each sibling group (local)
                for parent_key, siblings in siblings_by_parent.items():
                    if len(siblings) < 2:
                        continue
                    local_total_sibling_groups += 1
                    
                    # Use cached sibling similarities
                    upper_triangle = compute_and_cache_sibling_similarities(parent_key, siblings, code_embeddings)
                    
                    if upper_triangle is not None:
                        local_groups_with_embeddings += 1
                        local_total_pairs += len(upper_triangle)
                        local_duplicate_count += sum(1 for s in upper_triangle if s > 0.85)
            
            local_duplicate_penalty = local_duplicate_count / local_total_pairs if local_total_pairs > 0 else 0.0
            local_duplication_coverage = local_groups_with_embeddings / local_total_sibling_groups if local_total_sibling_groups > 0 else 0.0
            
            # 4. Global Duplication Penalty (fraction of level-wise topic pairs with cosine > 0.85)
            global_duplicate_count = 0
            global_total_pairs = 0
            global_total_level_nodes = 0
            global_levels_with_embeddings = 0
            
            for level, level_nodes in nodes_by_level.items():
                if level == 0 or len(level_nodes) < 2:
                    continue
                global_total_level_nodes += len(level_nodes)
                # Get embeddings for all nodes at this level
                level_embeddings = []
                for node in level_nodes:
                    if node.key in code_embeddings:
                        level_embeddings.append((node.key, code_embeddings[node.key]))
                
                if len(level_embeddings) >= 2:
                    global_levels_with_embeddings += 1
                    # Compute pairwise similarities for all nodes at this level
                    keys, embs = zip(*level_embeddings)
                    emb_matrix = np.array(embs, dtype=np.float32)
                    similarities = cosine_similarity(emb_matrix)
                    upper_triangle = similarities[np.triu_indices(len(similarities), k=1)]
                    global_total_pairs += len(upper_triangle)
                    global_duplicate_count += sum(1 for s in upper_triangle if s > 0.85)
                    # Clear intermediate arrays to free memory immediately
                    del emb_matrix, similarities
                    del level_embeddings  # Clear the list
                    upper_triangle = None
            
            global_duplicate_penalty = global_duplicate_count / global_total_pairs if global_total_pairs > 0 else 0.0
            global_duplication_coverage = sum(1 for level, level_nodes in nodes_by_level.items() 
                                             if level > 0 and len(level_nodes) >= 2 and 
                                             any(node.key in code_embeddings for node in level_nodes)) / \
                                         max(1, sum(1 for level, level_nodes in nodes_by_level.items() 
                                                   if level > 0 and len(level_nodes) >= 2))
            
            # 5. Intra-Level Diversity (1 - mean cosine) - REUSE cached sibling similarities
            level_diversities = []
            diversity_total_groups = 0
            diversity_groups_with_embeddings = 0
            
            for level, level_nodes in nodes_by_level.items():
                if level == 0:  # Skip root level
                    continue
                # Group siblings by parent
                siblings_by_parent = defaultdict(list)
                for node in level_nodes:
                    if node.parent and node.parent.key != 'root':
                        siblings_by_parent[node.parent.key].append(node)
                
                # Compute diversity for each sibling group
                for parent_key, siblings in siblings_by_parent.items():
                    if len(siblings) < 2:
                        continue
                    diversity_total_groups += 1
                    
                    # REUSE cached sibling similarities (computed in step 3)
                    upper_triangle = compute_and_cache_sibling_similarities(parent_key, siblings, code_embeddings)
                    
                    if upper_triangle is not None:
                        diversity_groups_with_embeddings += 1
                        mean_sim = float(np.mean(upper_triangle))
                        diversity = 1.0 - mean_sim
                        level_diversities.append(diversity)
            
            intra_level_diversity = float(np.mean(level_diversities)) if level_diversities else 0.0
            diversity_coverage = diversity_groups_with_embeddings / diversity_total_groups if diversity_total_groups > 0 else 0.0
            
            # 6. Inter-Level Differentiation (Granularity Δ) - REUSE cached sibling similarities
            # Mean sibling similarity (reuse from cache)
            sibling_similarities = []
            for cache_key, upper_triangle in sibling_similarity_cache.items():
                sibling_similarities.extend([float(s) for s in upper_triangle])
            
            mean_sibling_similarity = float(np.mean(sibling_similarities)) if sibling_similarities else 0.0
            inter_level_granularity = parent_child_similarity - mean_sibling_similarity
            
            # Clear sibling similarity cache after use (no longer needed)
            del sibling_similarity_cache
            sibling_similarities = None
            
            # 7. Net Purity - Two versions:
            #    a) Label-to-Content: Parent label vs child documents (content alignment)
            #    b) Label-to-Label: Parent label vs child label (structural alignment)
            net_purity_content_scores = []  # Parent label vs child documents
            net_purity_label_scores = []    # Parent label vs child label
            total_parent_child_pairs_for_purity = 0
            
            for node in nodes:
                if node.parent and node.parent.key != 'root':
                    parent_key = node.parent.key
                    child_key = node.key
                    total_parent_child_pairs_for_purity += 1
                    
                    # Get parent code embedding
                    if parent_key in code_embeddings:
                        parent_emb = code_embeddings[parent_key]
                        
                        # Version A: Parent label vs child documents (content-based)
                        if child_key in document_centroids:
                            child_centroid = document_centroids[child_key]
                            sim = cosine_similarity(
                                parent_emb.reshape(1, -1),
                                child_centroid.reshape(1, -1)
                            )[0][0]
                            net_purity_content_scores.append(float(sim))
                        
                        # Version B: Parent label vs child label (structural)
                        if child_key in code_embeddings:
                            child_emb = code_embeddings[child_key]
                            sim = cosine_similarity(
                                parent_emb.reshape(1, -1),
                                child_emb.reshape(1, -1)
                            )[0][0]
                            net_purity_label_scores.append(float(sim))
            
            # Use content-based version as primary (original metric)
            # Label-to-label is same as parent-child similarity, so we'll use content version
            net_purity = float(np.mean(net_purity_content_scores)) if net_purity_content_scores else 0.0
            net_purity_label = float(np.mean(net_purity_label_scores)) if net_purity_label_scores else 0.0
            net_purity_coverage = len(net_purity_content_scores) / total_parent_child_pairs_for_purity if total_parent_child_pairs_for_purity > 0 else 0.0
            
            # MEMORY OPTIMIZATION: Clear large data structures after last use
            # code_embeddings and document_centroids are no longer needed after net purity
            # code_to_documents is also no longer needed (only used for centroids)
            # These can be hundreds of MB for large hierarchies
            del code_embeddings
            del document_centroids
            del code_to_documents
            
            # Clear other large intermediate structures
            del codes_with_docs
            del nodes_by_level
            
            # Force garbage collection to free memory immediately
            import gc
            gc.collect()
            
            # 8. Composite Hierarchical Quality Score
            # Q_hier = α*C_embed + β*(1-D_dup) + γ*D_intra + δ*S_pc
            # Where: C_embed = coherence, D_dup = duplicate penalty (use global), D_intra = diversity, S_pc = parent-child similarity
            # Weights: α=0.3, β=0.3, γ=0.2, δ=0.2 (normalized to sum to 1.0)
            alpha, beta, gamma, delta = 0.3, 0.3, 0.2, 0.2
            
            # Ensure all components are in [0, 1] range before combining
            coherence_norm = max(0.0, min(1.0, embedding_coherence))
            duplication_norm = max(0.0, min(1.0, global_duplicate_penalty))
            diversity_norm = max(0.0, min(1.0, intra_level_diversity))
            parent_child_norm = max(0.0, min(1.0, parent_child_similarity))
            
            composite_score = (
                alpha * coherence_norm +
                beta * (1.0 - duplication_norm) +
                gamma * diversity_norm +
                delta * parent_child_norm
            )
            # Normalize to 0-1 range (already should be, but ensure)
            composite_score = max(0.0, min(1.0, composite_score))
            
            
            # 9. Generate comprehensive metrics summary
            metrics_summary = self._generate_metrics_summary({
                'embedding_coherence': embedding_coherence,
                'parent_child_similarity': parent_child_similarity,
                'local_duplicate_penalty': local_duplicate_penalty,
                'global_duplicate_penalty': global_duplicate_penalty,
                'intra_level_diversity': intra_level_diversity,
                'inter_level_granularity': inter_level_granularity,
                'net_purity': net_purity,
                'net_purity_label': net_purity_label,
                'composite_quality_score': composite_score
            })
            
            
            # Calculate overall coverage statistics (using cached counts)
            total_codes_count = len(nodes)
            coverage_stats = {
                'total_codes': total_codes_count,
                'codes_with_descriptions': codes_with_embeddings_count,
                'codes_with_documents': codes_with_docs_count,
                'codes_with_both': codes_with_both_count,
                'description_coverage': codes_with_embeddings_count / total_codes_count if total_codes_count > 0 else 0.0,
                'document_coverage': codes_with_docs_count / total_codes_count if total_codes_count > 0 else 0.0,
                'coherence_coverage': multi_doc_count / codes_with_docs_count if codes_with_docs_count > 0 else 0.0,
                'single_document_codes': single_doc_count,
                'parent_child_coverage': parent_child_coverage,
                'local_duplication_coverage': local_duplication_coverage,
                'diversity_coverage': diversity_coverage,
                'net_purity_coverage': net_purity_coverage
            }
            
            # Clear nodes list and other structures after computing stats (no longer needed)
            del nodes
            del all_nodes
            
            return {
                'embedding_coherence': round(embedding_coherence, 4),
                'parent_child_similarity': round(parent_child_similarity, 4),
                'local_duplicate_penalty': round(local_duplicate_penalty, 4),
                'global_duplicate_penalty': round(global_duplicate_penalty, 4),
                'intra_level_diversity': round(intra_level_diversity, 4),
                'inter_level_granularity': round(inter_level_granularity, 4),
                'net_purity': round(net_purity, 4),
                'net_purity_label': round(net_purity_label, 4),  # Label-to-label version
                'composite_quality_score': round(composite_score, 4),
                'metrics_summary': metrics_summary,
                'coverage_stats': coverage_stats,
                'max_depth': max_depth,
                'total_codes': total_codes_count,
                'codes_with_documents': codes_with_docs_count
            }
            
        except Exception as e:
            error_msg = f"Error computing metrics: {e}"
            print(f"[METRICS] {error_msg}")
            import traceback
            traceback_str = traceback.format_exc()
            print(f"[METRICS] Traceback:\n{traceback_str}")
            # Re-raise to be caught by the calling code
            raise ValueError(f"{error_msg}\n{traceback_str}")


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
    
    def extract_documents_per_code(self, xml_path: str) -> Dict[str, List[str]]:
        """
        Extract all verbatim documents assigned to each code.
        Returns mapping: code_key -> List[verbatim_text]
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Map code_key -> list of verbatims
        code_to_documents = defaultdict(list)
        
        # Get codebook to filter out invalid codes
        codebook = {}
        for codebook_elem in root.findall('.//CodeBook'):
            for code in codebook_elem.findall('.//CodeBookCode'):
                key = self._get_text(code, 'CBCKey')
                if key:
                    description = self._get_text(code, 'CBCDescription')
                    desc_norm = (description or '').strip().lower()
                    # Skip "Uncoded Segment" placeholder codes
                    if not ("uncoded" in desc_norm and "segment" in desc_norm):
                        codebook[key] = description
        
        # Extract responses and map to codes
        for question in root.findall('.//Question'):
            question_type = self._get_text(question, 'QuestionType', '0')
            
            # Only process open-ended questions (QuestionType = 0)
            if question_type != '0':
                continue
            
            for response in question.findall('.//Response'):
                verbatim = self._get_text(response, 'DROVerbatim', '').strip()
                if not verbatim:
                    continue
                
                # Get all codes assigned to this response
                codes = set()
                for resp_code in response.findall('.//ResponseCode'):
                    cbc_key = self._get_text(resp_code, 'DCCBCKey')
                    if cbc_key and cbc_key in codebook:
                        codes.add(cbc_key)
                
                # Add verbatim to all assigned codes
                for code_key in codes:
                    code_to_documents[code_key].append(verbatim)
        
        return dict(code_to_documents)
    
    def evaluate(self, progress_callback=None) -> Dict:
        """
        Extract hierarchy structures for visual comparison.
        
        Args:
            progress_callback: Deprecated - no longer used (kept for backward compatibility)
        """
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
        
        # Compute semantic metrics (no progress tracking)
        metrics_evaluator_benchmark = HierarchyMetricsEvaluator(progress_callback=None)
        benchmark_metrics = metrics_evaluator_benchmark.compute_metrics(
            self.benchmark_tree, 
            self.benchmark_path
        )
        
        metrics_evaluator_model = HierarchyMetricsEvaluator(progress_callback=None)
        model_metrics = metrics_evaluator_model.compute_metrics(
            self.model_tree,
            self.model_path
        )
        
        
        # Compute comparison deltas
        comparison = {}
        if benchmark_metrics and model_metrics:
            comparison = {
                'delta_coherence': model_metrics.get('embedding_coherence', 0) - benchmark_metrics.get('embedding_coherence', 0),
                'delta_parent_child': model_metrics.get('parent_child_similarity', 0) - benchmark_metrics.get('parent_child_similarity', 0),
                'delta_local_duplicate_penalty': benchmark_metrics.get('local_duplicate_penalty', 0) - model_metrics.get('local_duplicate_penalty', 0),
                'delta_global_duplicate_penalty': benchmark_metrics.get('global_duplicate_penalty', 0) - model_metrics.get('global_duplicate_penalty', 0),
                'delta_diversity': model_metrics.get('intra_level_diversity', 0) - benchmark_metrics.get('intra_level_diversity', 0),
                'delta_granularity': model_metrics.get('inter_level_granularity', 0) - benchmark_metrics.get('inter_level_granularity', 0),
                'delta_net_purity': model_metrics.get('net_purity', 0) - benchmark_metrics.get('net_purity', 0),
                'delta_net_purity_label': model_metrics.get('net_purity_label', 0) - benchmark_metrics.get('net_purity_label', 0),
                'delta_composite_quality': model_metrics.get('composite_quality_score', 0) - benchmark_metrics.get('composite_quality_score', 0),
            }
        
        
        return {
            'benchmark': {
                'tree': self.benchmark_tree.to_dict() if self.benchmark_tree else {},
                'metrics': benchmark_metrics or {}
            },
            'model': {
                'tree': self.model_tree.to_dict() if self.model_tree else {},
                'metrics': model_metrics or {}
            },
            'comparison': comparison
        }
