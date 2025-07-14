"""
XAI Benchmark - Metrics Module
Implementa le 4 metriche di valutazione XAI dal paper:
1. Human-reasoning Agreement (HA) || temporaneamente disabilitato 
2. Robustness 
3. Consistency
4. Contrastivity
"""

import logging
import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score
import copy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HumanAgreementMetric:
    """
    Human-reasoning Agreement (HA) Metric
    Misura l'allineamento tra spiegazioni XAI e annotazioni umane usando Mean Average Precision (MAP)
    """
    
    def __init__(self):
        self.name = "Human-reasoning Agreement"
        self.abbreviation = "HA"
    
    def compute_average_precision(self, 
                                 xai_tokens: List[str], 
                                 xai_importances: List[float],
                                 human_tokens: List[str]) -> float:
        """
        Calcola Average Precision per una singola istanza
        
        Args:
            xai_tokens: Token dalla spiegazione XAI
            xai_importances: Importanze dei token XAI
            human_tokens: Token importanti secondo annotazione umana
            
        Returns:
            Average Precision score
        """
        if not xai_tokens or not human_tokens:
            return 0.0
        
        # Ordina token XAI per importanza (decrescente)
        token_importance_pairs = list(zip(xai_tokens, xai_importances))
        token_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Crea ranking XAI
        ranked_xai_tokens = [token for token, _ in token_importance_pairs]
        
        # Normalizza token per confronto (lowercase, rimuovi prefissi)
        def normalize_token(token):
            return token.replace('##', '').replace('Ġ', '').lower().strip()
        
        normalized_human = set(normalize_token(token) for token in human_tokens)
        
        # Calcola AP
        relevant_found = 0
        ap_sum = 0.0
        
        for k, xai_token in enumerate(ranked_xai_tokens, 1):
            normalized_xai = normalize_token(xai_token)
            
            if normalized_xai in normalized_human:
                relevant_found += 1
                precision_at_k = relevant_found / k
                ap_sum += precision_at_k
        
        if relevant_found == 0:
            return 0.0
        
        return ap_sum / len(normalized_human)
    
    def compute_map(self, 
                   xai_explanations: List[Dict[str, Any]], 
                   human_annotations: List[List[str]]) -> float:
        """
        Calcola Mean Average Precision su un set di istanze
        
        Args:
            xai_explanations: Lista di spiegazioni XAI
            human_annotations: Lista di annotazioni umane (token importanti)
            
        Returns:
            Mean Average Precision score
        """
        if len(xai_explanations) != len(human_annotations):
            raise ValueError("Numero di spiegazioni XAI e annotazioni umane deve essere uguale")
        
        ap_scores = []
        
        for xai_exp, human_tokens in zip(xai_explanations, human_annotations):
            ap = self.compute_average_precision(
                xai_exp['tokens'],
                xai_exp['token_importances'], 
                human_tokens
            )
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def evaluate(self, 
                xai_explanations: List[Dict[str, Any]], 
                human_annotations: List[List[str]]) -> Dict[str, float]:
        """
        Valuta Human-reasoning Agreement
        
        Returns:
            Dict con MAP score e statistiche
        """
        map_score = self.compute_map(xai_explanations, human_annotations)
        
        # Calcola AP per ogni istanza per statistiche
        ap_scores = []
        for xai_exp, human_tokens in zip(xai_explanations, human_annotations):
            ap = self.compute_average_precision(
                xai_exp['tokens'],
                xai_exp['token_importances'],
                human_tokens
            )
            ap_scores.append(ap)
        
        return {
            'map': map_score,
            'mean_ap': np.mean(ap_scores),
            'std_ap': np.std(ap_scores),
            'min_ap': np.min(ap_scores),
            'max_ap': np.max(ap_scores),
            'num_instances': len(ap_scores)
        }


class RobustnessMetric:
    """
    Robustness Metric
    Misura la stabilità delle spiegazioni sotto perturbazioni dell'input
    """
    
    def __init__(self, perturbation_ratio: float = 0.1):
        self.name = "Robustness"
        self.abbreviation = "R"
        self.perturbation_ratio = perturbation_ratio
    
    def create_perturbations(self, text: str, num_perturbations: int = 5) -> List[str]:
        """
        Crea perturbazioni del testo originale
        
        Args:
            text: Testo originale
            num_perturbations: Numero di perturbazioni da creare
            
        Returns:
            Lista di testi perturbati
        """
        words = text.split()
        if len(words) < 2:
            return [text] * num_perturbations
        
        perturbations = []
        num_words_to_change = max(1, int(len(words) * self.perturbation_ratio))
        
        for _ in range(num_perturbations):
            perturbed_words = words.copy()
            
            # Seleziona parole casuali da modificare
            indices_to_change = random.sample(range(len(words)), 
                                            min(num_words_to_change, len(words)))
            
            for idx in indices_to_change:
                # Strategie di perturbazione
                strategy = random.choice(['mask', 'synonym', 'delete'])
                
                if strategy == 'mask':
                    perturbed_words[idx] = '[MASK]'
                elif strategy == 'synonym':
                    # Semplice sostituzione con parole comuni
                    synonyms = ['good', 'bad', 'nice', 'great', 'poor', 'excellent', 'terrible']
                    perturbed_words[idx] = random.choice(synonyms)
                elif strategy == 'delete' and len(perturbed_words) > 1:
                    perturbed_words.pop(idx)
                    break  # Evita problemi con indici dopo deletion
            
            perturbations.append(' '.join(perturbed_words))
        
        return perturbations
    
    def compute_explanation_distance(self, 
                                   explanation1: Dict[str, Any], 
                                   explanation2: Dict[str, Any]) -> float:
        """
        Calcola distanza tra due spiegazioni
        
        Args:
            explanation1: Prima spiegazione
            explanation2: Seconda spiegazione
            
        Returns:
            Distanza media tra importanze dei token
        """
        tokens1 = explanation1['tokens']
        importances1 = explanation1['token_importances']
        tokens2 = explanation2['tokens']
        importances2 = explanation2['token_importances']
        
        # Crea dizionari token -> importanza
        dict1 = {token: imp for token, imp in zip(tokens1, importances1)}
        dict2 = {token: imp for token, imp in zip(tokens2, importances2)}
        
        # Trova token comuni
        common_tokens = set(dict1.keys()) & set(dict2.keys())
        
        if not common_tokens:
            return 1.0  # Massima distanza se nessun token comune
        
        # Calcola differenze per token comuni
        differences = []
        for token in common_tokens:
            diff = abs(dict1[token] - dict2[token])
            differences.append(diff)
        
        return np.mean(differences)
    
    def evaluate(self, 
                explanations: List[Dict[str, Any]], 
                texts: List[str],
                explainer_function: callable) -> Dict[str, float]:
        """
        Valuta robustness delle spiegazioni
        
        Args:
            explanations: Spiegazioni originali
            texts: Testi originali
            explainer_function: Funzione per generare spiegazioni
            
        Returns:
            Dict con metriche di robustness
        """
        all_distances = []
        
        for original_explanation, original_text in zip(explanations, texts):
            # Crea perturbazioni
            perturbations = self.create_perturbations(original_text)
            
            # Genera spiegazioni per perturbazioni
            perturbation_distances = []
            
            for perturbed_text in perturbations:
                try:
                    perturbed_explanation = explainer_function(perturbed_text)
                    distance = self.compute_explanation_distance(
                        original_explanation, perturbed_explanation
                    )
                    perturbation_distances.append(distance)
                except Exception as e:
                    logger.warning(f"Errore nella perturbazione: {str(e)}")
                    continue
            
            if perturbation_distances:
                avg_distance = np.mean(perturbation_distances)
                all_distances.append(avg_distance)
        
        if not all_distances:
            return {'mad': 1.0, 'std': 0.0, 'min': 1.0, 'max': 1.0}
        
        return {
            'mad': np.mean(all_distances),  # Mean Average Distance (lower is better)
            'std': np.std(all_distances),
            'min': np.min(all_distances),
            'max': np.max(all_distances),
            'num_instances': len(all_distances)
        }


class ConsistencyMetric:
    """
    Consistency Metric
    Misura la consistenza delle spiegazioni tra modelli con architetture simili
    """
    
    def __init__(self):
        self.name = "Consistency"
        self.abbreviation = "Cn"
    
    def compute_attention_distance(self, 
                                 attention_weights1: torch.Tensor, 
                                 attention_weights2: torch.Tensor) -> float:
        """
        Calcola distanza tra attention weights di due modelli
        
        Args:
            attention_weights1: Attention del primo modello
            attention_weights2: Attention del secondo modello
            
        Returns:
            Distanza coseno tra attention weights
        """
        # Flatten e converte a numpy
        att1_flat = attention_weights1.flatten().cpu().numpy()
        att2_flat = attention_weights2.flatten().cpu().numpy()
        
        # Gestisci dimensioni diverse
        min_len = min(len(att1_flat), len(att2_flat))
        att1_flat = att1_flat[:min_len]
        att2_flat = att2_flat[:min_len]
        
        # Calcola distanza coseno 
        try:
            distance = cosine(att1_flat, att2_flat)
            return distance if not np.isnan(distance) else 1.0
        except:
            return 1.0
    
    def compute_explanation_distance(self, 
                                   explanation1: Dict[str, Any], 
                                   explanation2: Dict[str, Any]) -> float:
        """
        Calcola distanza tra spiegazioni di due modelli
        
        Args:
            explanation1: Spiegazione del primo modello
            explanation2: Spiegazione del secondo modello
            
        Returns:
            Distanza tra le spiegazioni
        """
        tokens1 = explanation1['tokens']
        importances1 = explanation1['token_importances']
        tokens2 = explanation2['tokens']
        importances2 = explanation2['token_importances']
        
        # Allinea token e importanze
        dict1 = {token: imp for token, imp in zip(tokens1, importances1)}
        dict2 = {token: imp for token, imp in zip(tokens2, importances2)}
        
        all_tokens = set(dict1.keys()) | set(dict2.keys())
        
        if not all_tokens:
            return 0.0
        
        # Crea vettori allineati
        vec1 = [dict1.get(token, 0.0) for token in all_tokens]
        vec2 = [dict2.get(token, 0.0) for token in all_tokens]
        
        # Calcola correlazione di Spearman
        try:
            correlation, _ = spearmanr(vec1, vec2)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def evaluate(self, 
                explanations_model1: List[Dict[str, Any]], 
                explanations_model2: List[Dict[str, Any]],
                attention_weights1: Optional[List[torch.Tensor]] = None,
                attention_weights2: Optional[List[torch.Tensor]] = None) -> Dict[str, float]:
        """
        Valuta consistency tra due set di spiegazioni
        
        Args:
            explanations_model1: Spiegazioni del primo modello
            explanations_model2: Spiegazioni del secondo modello
            attention_weights1: Attention weights del primo modello (opzionale)
            attention_weights2: Attention weights del secondo modello (opzionale)
            
        Returns:
            Dict con metriche di consistency
        """
        if len(explanations_model1) != len(explanations_model2):
            raise ValueError("Numero di spiegazioni deve essere uguale per entrambi i modelli")
        
        # Calcola distanze delle spiegazioni
        explanation_correlations = []
        
        for exp1, exp2 in zip(explanations_model1, explanations_model2):
            correlation = self.compute_explanation_distance(exp1, exp2)
            explanation_correlations.append(correlation)
        
        results = {
            'explanation_correlation': np.mean(explanation_correlations),
            'explanation_std': np.std(explanation_correlations),
            'num_instances': len(explanation_correlations)
        }
        
        # Se disponibili, calcola anche distanze attention
        if attention_weights1 is not None and attention_weights2 is not None:
            attention_distances = []
            
            for att1, att2 in zip(attention_weights1, attention_weights2):
                distance = self.compute_attention_distance(att1, att2)
                attention_distances.append(distance)
            
            results.update({
                'attention_distance': np.mean(attention_distances),
                'attention_std': np.std(attention_distances),
                'overall_consistency': np.mean([
                    np.mean(explanation_correlations),
                    1.0 - np.mean(attention_distances)  # Converti distanza in similarità
                ])
            })
        else:
            results['overall_consistency'] = np.mean(explanation_correlations)
        
        return results


class ContrastivityMetric:
    """
    Contrastivity Metric
    Misura quanto bene un metodo XAI distingue tra diverse classi
    """
    
    def __init__(self):
        self.name = "Contrastivity"
        self.abbreviation = "Ct"
    
    def compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calcola KL divergence tra due distribuzioni
        
        Args:
            p: Prima distribuzione
            q: Seconda distribuzione
            
        Returns:
            KL divergence
        """
        # Evita log(0) aggiungendo piccolo epsilon
        epsilon = 1e-8
        p = np.array(p) + epsilon
        q = np.array(q) + epsilon
        
        # Normalizza per assicurare che siano distribuzioni di probabilità
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log(p / q))
    
    def prepare_importance_distributions(self, 
                                       explanations: List[Dict[str, Any]], 
                                       class_labels: List[int]) -> Dict[int, List[np.ndarray]]:
        """
        Prepara distribuzioni di importanza per classe
        
        Args:
            explanations: Lista di spiegazioni
            class_labels: Labels di classe corrispondenti
            
        Returns:
            Dict {class: [distributions]}
        """
        class_distributions = {}
        
        for explanation, label in zip(explanations, class_labels):
            if label not in class_distributions:
                class_distributions[label] = []
            
            # Normalizza importanze per creare distribuzione
            importances = np.array(explanation['token_importances'])
            
            # Converti in valori positivi (prendi valore assoluto)
            importances = np.abs(importances)
            
            if np.sum(importances) > 0:
                # Normalizza per ottenere distribuzione
                distribution = importances / np.sum(importances)
            else:
                # Distribuzione uniforme se tutte le importanze sono zero
                distribution = np.ones(len(importances)) / len(importances)
            
            class_distributions[label].append(distribution)
        
        return class_distributions
    
    def evaluate(self, 
                explanations: List[Dict[str, Any]], 
                class_labels: List[int]) -> Dict[str, float]:
        """
        Valuta contrastivity delle spiegazioni
        
        Args:
            explanations: Lista di spiegazioni
            class_labels: Labels di classe corrispondenti
            
        Returns:
            Dict con metriche di contrastivity
        """
        if len(explanations) != len(class_labels):
            raise ValueError("Numero di spiegazioni e labels deve essere uguale")
        
        # Prepara distribuzioni per classe
        class_distributions = self.prepare_importance_distributions(explanations, class_labels)
        
        if len(class_distributions) < 2:
            logger.warning("Meno di 2 classi trovate per contrastivity")
            return {'kl_divergence': 0.0, 'mean_kl': 0.0, 'std_kl': 0.0}
        
        # Calcola KL divergence tra tutte le coppie di classi
        kl_divergences = []
        class_pairs = []
        
        classes = list(class_distributions.keys())
        
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes):
                if i >= j:  # Evita duplicati e auto-confronti
                    continue
                
                distributions1 = class_distributions[class1]
                distributions2 = class_distributions[class2]
                
                # Calcola KL divergence per tutte le coppie di distribuzioni
                pair_kls = []
                
                for dist1 in distributions1:
                    for dist2 in distributions2:
                        # Assicura stessa lunghezza
                        min_len = min(len(dist1), len(dist2))
                        if min_len == 0:
                            continue
                        
                        d1_aligned = dist1[:min_len]
                        d2_aligned = dist2[:min_len]
                        
                        # KL divergence simmetrica
                        kl1 = self.compute_kl_divergence(d1_aligned, d2_aligned)
                        kl2 = self.compute_kl_divergence(d2_aligned, d1_aligned)
                        symmetric_kl = (kl1 + kl2) / 2
                        
                        pair_kls.append(symmetric_kl)
                
                if pair_kls:
                    avg_kl = np.mean(pair_kls)
                    kl_divergences.append(avg_kl)
                    class_pairs.append((class1, class2))
        
        if not kl_divergences:
            return {'kl_divergence': 0.0, 'mean_kl': 0.0, 'std_kl': 0.0}
        
        return {
            'kl_divergence': np.mean(kl_divergences),  # Higher is better (more contrastive)
            'mean_kl': np.mean(kl_divergences),
            'std_kl': np.std(kl_divergences),
            'max_kl': np.max(kl_divergences),
            'min_kl': np.min(kl_divergences),
            'num_class_pairs': len(class_pairs),
            'class_pairs': class_pairs
        }


class MetricsManager:
    """Gestisce tutte le metriche di valutazione XAI"""
    
    def __init__(self):
        self.metrics = {
            'human_agreement': HumanAgreementMetric(),
            'robustness': RobustnessMetric(),
            'consistency': ConsistencyMetric(),
            'contrastivity': ContrastivityMetric()
        }
    
    def evaluate_all_metrics(self, 
                           xai_explanations: List[Dict[str, Any]],
                           texts: Optional[List[str]] = None,
                           explainer_function: Optional[callable] = None,
                           explanations_model2: Optional[List[Dict[str, Any]]] = None,
                           class_labels: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Valuta tutte le metriche disponibili
        
        Args:
            xai_explanations: Spiegazioni XAI primarie
            texts: Testi originali per Robustness
            explainer_function: Funzione explainer per Robustness
            explanations_model2: Spiegazioni secondo modello per Consistency
            class_labels: Labels di classe per Contrastivity
            
        Returns:
            Dict con risultati di tutte le metriche
        """
        results = {}
        
        # NOTA: Human Agreement temporaneamente disabilitato
        # Richiede annotazioni umane che verranno implementate successivamente
        
        # Robustness
        if texts is not None and explainer_function is not None:
            try:
                robustness_results = self.metrics['robustness'].evaluate(
                    xai_explanations, texts, explainer_function
                )
                results['robustness'] = robustness_results
                logger.info(f"SUCCESS: Robustness - MAD: {robustness_results['mad']:.4f}")
            except Exception as e:
                logger.warning(f"FAILED: Robustness - {str(e)}")
                results['robustness'] = None
        
        # Consistency
        if explanations_model2 is not None:
            try:
                consistency_results = self.metrics['consistency'].evaluate(
                    xai_explanations, explanations_model2
                )
                results['consistency'] = consistency_results
                logger.info(f"SUCCESS: Consistency - Correlation: {consistency_results['explanation_correlation']:.4f}")
            except Exception as e:
                logger.warning(f"FAILED: Consistency - {str(e)}")
                results['consistency'] = None
        
        # Contrastivity
        if class_labels is not None:
            try:
                contrastivity_results = self.metrics['contrastivity'].evaluate(
                    xai_explanations, class_labels
                )
                results['contrastivity'] = contrastivity_results
                logger.info(f"SUCCESS: Contrastivity - KL: {contrastivity_results['kl_divergence']:.4f}")
            except Exception as e:
                logger.warning(f"FAILED: Contrastivity - {str(e)}")
                results['contrastivity'] = None
        
        return results
    
    def get_metrics_summary(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Estrae summary con score principali
        
        Args:
            results: Risultati delle metriche
            
        Returns:
            Dict con score principali
        """
        summary = {}
        
        # NOTA: Human Agreement temporaneamente rimosso
        
        if results.get('robustness') is not None:
            summary['robustness_mad'] = results['robustness']['mad']
        
        if results.get('consistency') is not None:
            summary['consistency_correlation'] = results['consistency']['explanation_correlation']
        
        if results.get('contrastivity') is not None:
            summary['contrastivity_kl'] = results['contrastivity']['kl_divergence']
        
        return summary


def test_metrics():
    """Test rapido delle metriche"""
    try:
        print("\nTEST: Inizializzazione metriche")
        
        # Crea dati di test
        dummy_explanations = [
            {
                'tokens': ['this', 'movie', 'is', 'great'],
                'token_importances': [0.1, 0.3, 0.1, 0.8],
                'predicted_class': 1
            },
            {
                'tokens': ['terrible', 'film', 'bad'],
                'token_importances': [0.9, 0.2, 0.7],
                'predicted_class': 0
            }
        ]
        
        class_labels = [1, 0]
        
        # Test Manager
        manager = MetricsManager()
        print("SUCCESS: MetricsManager inizializzato")
        
        # NOTA: Human Agreement test rimosso temporaneamente
        
        # Test Contrastivity
        ct_metric = manager.metrics['contrastivity']
        ct_results = ct_metric.evaluate(dummy_explanations, class_labels)
        print(f"SUCCESS: Contrastivity - KL: {ct_results['kl_divergence']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Test metriche fallito: {str(e)}")
        return False


if __name__ == "__main__":
    # Demo
    test_metrics()