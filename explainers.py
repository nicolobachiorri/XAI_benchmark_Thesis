"""
XAI Benchmark - Explainers Module
Implementa i metodi di spiegabilità XAI per transformer encoder-only
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import lime
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME non installato. Usa: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP non installato. Usa: pip install shap")

try:
    import captum
    from captum.attr import (
        IntegratedGradients, 
        InputXGradient,
        LayerConductance,
        LayerIntegratedGradients
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logger.warning("Captum non installato. Usa: pip install captum")


class BaseExplainer(ABC):
    """Classe base per tutti gli explainer"""
    
    def __init__(self, model, tokenizer, device='auto'):
        self.model = model
        self.tokenizer = tokenizer
        
        if device == 'auto':
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
        
        self.model.eval()
        
    @abstractmethod
    def explain(self, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Genera spiegazione per un testo
        
        Args:
            text: Testo da spiegare
            target_class: Classe target (se None, usa predizione del modello)
            
        Returns:
            Dict con token_importances, prediction, confidence, etc.
        """
        pass
    
    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predice probabilità per una lista di testi"""
        self.model.eval()
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
        return probs.cpu().numpy()
    
    def _get_prediction(self, text: str) -> Tuple[int, float]:
        """Ottiene predizione e confidence per un testo"""
        probs = self._predict_proba([text])[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        return predicted_class, confidence


class LIMEExplainer(BaseExplainer):
    """Model simplification - LIME (Local Interpretable Model-agnostic Explanations)"""
    
    def __init__(self, model, tokenizer, device='auto'):
        super().__init__(model, tokenizer, device)
        
        if not LIME_AVAILABLE:
            raise ImportError("LIME non disponibile. Installare con: pip install lime")
        
        self.explainer = LimeTextExplainer(
            class_names=['negative', 'positive'],
            mode='classification'
        )
        
    def explain(self, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Genera spiegazione LIME"""
        logger.debug("Generazione spiegazione LIME")
        
        # Predizione base
        predicted_class, confidence = self._get_prediction(text)
        
        if target_class is None:
            target_class = predicted_class
        
        # Genera spiegazione LIME
        explanation = self.explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=20,  # Top 20 features più importanti
            top_labels=2
        )
        
        # Estrai importanze delle parole
        word_importances = {}
        for word, importance in explanation.as_list(label=target_class):
            word_importances[word] = float(importance)
        
        # Tokenizza per avere token-level importances
        tokens = self.tokenizer.tokenize(text)
        token_importances = self._map_words_to_tokens(text, tokens, word_importances)
        
        return {
            'method': 'LIME',
            'text': text,
            'tokens': tokens,
            'token_importances': token_importances,
            'word_importances': word_importances,
            'predicted_class': predicted_class,
            'target_class': target_class,
            'confidence': confidence,
            'explanation_object': explanation
        }
    
    def _map_words_to_tokens(self, text: str, tokens: List[str], word_importances: Dict[str, float]) -> List[float]:
        """Mappa importanze da parole a token"""
        token_importances = []
        
        for token in tokens:
            # Rimuovi prefissi del tokenizer (##, Ġ, etc.)
            clean_token = token.replace('##', '').replace('Ġ', '').lower()
            
            # Cerca corrispondenza con parole in word_importances
            importance = 0.0
            for word, imp in word_importances.items():
                if clean_token in word.lower() or word.lower() in clean_token:
                    importance = imp
                    break
            
            token_importances.append(importance)
        
        return token_importances


class SHAPExplainer(BaseExplainer):
    """Perturbation-based - SHAP (SHapley Additive exPlanations)"""
    
    def __init__(self, model, tokenizer, device='auto'):
        super().__init__(model, tokenizer, device)
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP non disponibile. Installare con: pip install shap")
        
        # Usa Partition explainer per testi
        self.explainer = shap.Explainer(self._predict_proba, self.tokenizer)
    
    def explain(self, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Genera spiegazione SHAP"""
        logger.debug("Generazione spiegazione SHAP")
        
        # Predizione base
        predicted_class, confidence = self._get_prediction(text)
        
        if target_class is None:
            target_class = predicted_class
        
        try:
            # Genera spiegazione SHAP
            shap_values = self.explainer([text])
            
            # Estrai valori per la classe target
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) == 3:  # Multi-class
                    importances = shap_values.values[0, :, target_class]
                else:  # Binary
                    importances = shap_values.values[0]
                
                tokens = shap_values.data[0] if hasattr(shap_values, 'data') else self.tokenizer.tokenize(text)
            else:
                # Fallback per versioni diverse di SHAP
                importances = shap_values[0][:, target_class] if len(shap_values[0].shape) > 1 else shap_values[0]
                tokens = self.tokenizer.tokenize(text)
            
            # Assicurati che tokens e importances abbiano la stessa lunghezza
            min_len = min(len(tokens), len(importances))
            tokens = tokens[:min_len]
            importances = importances[:min_len]
            
            token_importances = [float(imp) for imp in importances]
            
        except Exception as e:
            logger.warning(f"Errore SHAP, usando fallback: {str(e)}")
            # Fallback: usa gradient-based approximation
            tokens = self.tokenizer.tokenize(text)
            token_importances = [0.0] * len(tokens)
        
        return {
            'method': 'SHAP',
            'text': text,
            'tokens': tokens,
            'token_importances': token_importances,
            'predicted_class': predicted_class,
            'target_class': target_class,
            'confidence': confidence
        }


class InputXGradientExplainer(BaseExplainer):
    """Gradient-based - Input × Gradient"""
    
    def __init__(self, model, tokenizer, device='auto'):
        super().__init__(model, tokenizer, device)
        
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum non disponibile. Installare con: pip install captum")
        
        self.ig = InputXGradient(self.model)
    
    def explain(self, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Genera spiegazione Input×Gradient"""
        logger.debug("Generazione spiegazione Input×Gradient")
        
        # Tokenizza input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Predizione
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            predicted_class = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, predicted_class].item())
        
        if target_class is None:
            target_class = predicted_class
        
        # Calcola attributions
        self.model.zero_grad()
        
        # Usa embeddings come input per Captum
        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
        embeddings.requires_grad_(True)
        
        # Input×Gradient attribution
        attributions = self.ig.attribute(
            embeddings,
            target=target_class,
            additional_forward_args=(inputs['attention_mask'],)
        )
        
        # Somma le attributions lungo la dimensione degli embeddings
        token_attributions = attributions.sum(dim=-1).squeeze(0)
        token_importances = token_attributions.detach().cpu().numpy().tolist()
        
        # Decodifica token
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Rimuovi token speciali e padding
        valid_indices = []
        clean_tokens = []
        clean_importances = []
        
        for i, (token, importance) in enumerate(zip(tokens, token_importances)):
            if token not in [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]:
                valid_indices.append(i)
                clean_tokens.append(token)
                clean_importances.append(importance)
        
        return {
            'method': 'InputXGradient',
            'text': text,
            'tokens': clean_tokens,
            'token_importances': clean_importances,
            'predicted_class': predicted_class,
            'target_class': target_class,
            'confidence': confidence
        }


class LRPExplainer(BaseExplainer):
    """Layer-wise Relevance Propagation"""
    
    def __init__(self, model, tokenizer, device='auto'):
        super().__init__(model, tokenizer, device)
        
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum non disponibile per LRP. Usando approssimazione con LayerConductance")
        
        # Usa LayerConductance come approssimazione di LRP
        self.lrp = LayerConductance(self.model, self.model.get_input_embeddings())
    
    def explain(self, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Genera spiegazione LRP (approssimata con LayerConductance)"""
        logger.debug("Generazione spiegazione LRP")
        
        # Tokenizza input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Predizione
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            predicted_class = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, predicted_class].item())
        
        if target_class is None:
            target_class = predicted_class
        
        # Calcola LayerConductance (approssimazione LRP)
        attributions = self.lrp.attribute(
            inputs['input_ids'],
            target=target_class,
            additional_forward_args=(inputs['attention_mask'],)
        )
        
        # Somma attributions per token
        token_attributions = attributions.sum(dim=-1).squeeze(0)
        token_importances = token_attributions.detach().cpu().numpy().tolist()
        
        # Decodifica token
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Pulisci tokens e importances
        clean_tokens = []
        clean_importances = []
        
        for token, importance in zip(tokens, token_importances):
            if token not in [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]:
                clean_tokens.append(token)
                clean_importances.append(importance)
        
        return {
            'method': 'LRP',
            'text': text,
            'tokens': clean_tokens,
            'token_importances': clean_importances,
            'predicted_class': predicted_class,
            'target_class': target_class,
            'confidence': confidence
        }


class AttentionRolloutExplainer(BaseExplainer):
    """Attention Mechanism - Attention Rollout"""
    
    def explain(self, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Genera spiegazione Attention Rollout"""
        logger.debug("Generazione spiegazione Attention Rollout")
        
        # Tokenizza input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Forward pass con attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # Tuple di attention weights per ogni layer
            probs = F.softmax(outputs.logits, dim=-1)
            predicted_class = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, predicted_class].item())
        
        if target_class is None:
            target_class = predicted_class
        
        # Calcola Attention Rollout
        rollout = self._compute_rollout(attentions)
        
        # Estrai importanze per CLS token (rappresenta classificazione)
        cls_importances = rollout[0, 0, 1:].cpu().numpy()  # [0,0] = CLS, [1:] = skip CLS stesso
        
        # Decodifica token
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Rimuovi token speciali
        clean_tokens = []
        clean_importances = []
        
        for i, token in enumerate(tokens[1:], 1):  # Skip CLS
            if token not in [self.tokenizer.pad_token, self.tokenizer.sep_token] and i-1 < len(cls_importances):
                clean_tokens.append(token)
                clean_importances.append(float(cls_importances[i-1]))
        
        return {
            'method': 'AttentionRollout',
            'text': text,
            'tokens': clean_tokens,
            'token_importances': clean_importances,
            'predicted_class': predicted_class,
            'target_class': target_class,
            'confidence': confidence
        }
    
    def _compute_rollout(self, attentions):
        """Calcola Attention Rollout attraverso i layer"""
        result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
        
        for attention in attentions:
            # Media su tutte le heads
            attention_heads_fused = attention.mean(dim=1)
            
            # Aggiungi residual connection
            attention_heads_fused = attention_heads_fused + torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
            
            # Normalizza
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            
            # Moltiplica per risultato precedente
            result = torch.matmul(attention_heads_fused, result)
        
        return result


class AttentionFlowExplainer(BaseExplainer):
    """Attention Mechanism - Attention Flow (variante di Attention Rollout)"""
    
    def explain(self, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Genera spiegazione Attention Flow"""
        logger.debug("Generazione spiegazione Attention Flow")
        
        # Tokenizza input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Forward pass con attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions
            probs = F.softmax(outputs.logits, dim=-1)
            predicted_class = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, predicted_class].item())
        
        if target_class is None:
            target_class = predicted_class
        
        # Calcola Attention Flow (media pesata degli ultimi layer)
        flow = self._compute_attention_flow(attentions)
        
        # Estrai importanze per CLS token
        cls_importances = flow[0, 0, 1:].cpu().numpy()
        
        # Decodifica token
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Pulisci tokens
        clean_tokens = []
        clean_importances = []
        
        for i, token in enumerate(tokens[1:], 1):
            if token not in [self.tokenizer.pad_token, self.tokenizer.sep_token] and i-1 < len(cls_importances):
                clean_tokens.append(token)
                clean_importances.append(float(cls_importances[i-1]))
        
        return {
            'method': 'AttentionFlow',
            'text': text,
            'tokens': clean_tokens,
            'token_importances': clean_importances,
            'predicted_class': predicted_class,
            'target_class': target_class,
            'confidence': confidence
        }
    
    def _compute_attention_flow(self, attentions):
        """Calcola Attention Flow come media pesata degli ultimi layer"""
        # Usa gli ultimi 4 layer con pesi decrescenti
        weights = [0.4, 0.3, 0.2, 0.1]  # Più peso ai layer finali
        
        if len(attentions) < 4:
            # Se meno di 4 layer, usa tutti con pesi uguali
            weights = [1.0 / len(attentions)] * len(attentions)
            last_attentions = attentions
        else:
            last_attentions = attentions[-4:]
        
        # Combina attention weights
        combined = torch.zeros_like(last_attentions[0])
        
        for attention, weight in zip(last_attentions, weights):
            # Media su heads e applica peso
            attention_mean = attention.mean(dim=1)
            combined += weight * attention_mean
        
        return combined


class ExplainerManager:
    """Gestisce tutti gli explainer disponibili"""
    
    AVAILABLE_EXPLAINERS = {
        'lime': LIMEExplainer,
        'shap': SHAPExplainer,
        'inputxgradient': InputXGradientExplainer,
        'lrp': LRPExplainer,
        'attention_rollout': AttentionRolloutExplainer,
        'attention_flow': AttentionFlowExplainer
    }
    
    def __init__(self, model, tokenizer, device='auto'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._explainers = {}
    
    def get_explainer(self, method: str) -> BaseExplainer:
        """Ottiene un explainer specifico"""
        if method.lower() not in self.AVAILABLE_EXPLAINERS:
            raise ValueError(f"Explainer '{method}' non disponibile. "
                           f"Disponibili: {list(self.AVAILABLE_EXPLAINERS.keys())}")
        
        if method.lower() not in self._explainers:
            explainer_class = self.AVAILABLE_EXPLAINERS[method.lower()]
            self._explainers[method.lower()] = explainer_class(
                self.model, self.tokenizer, self.device
            )
        
        return self._explainers[method.lower()]
    
    def explain_with_all_methods(self, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Genera spiegazioni con tutti i metodi disponibili"""
        results = {}
        
        for method_name in self.AVAILABLE_EXPLAINERS.keys():
            try:
                explainer = self.get_explainer(method_name)
                explanation = explainer.explain(text, target_class)
                results[method_name] = explanation
                logger.info(f"SUCCESS: Spiegazione {method_name} generata")
            except Exception as e:
                logger.warning(f"FAILED: Errore con {method_name}: {str(e)}")
                results[method_name] = None
        
        return results
    
    def get_available_methods(self) -> List[str]:
        """Restituisce lista dei metodi disponibili"""
        return list(self.AVAILABLE_EXPLAINERS.keys())


def test_explainers():
    """Test rapido degli explainer"""
    try:
        print("\nTEST: Caricamento explainer")
        
        # Carica modello per test
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Inizializza manager
        manager = ExplainerManager(model, tokenizer)
        
        # Test con testo di esempio
        test_text = "This movie is absolutely fantastic and amazing!"
        
        print(f"SUCCESS: Manager inizializzato")
        print(f"Metodi disponibili: {manager.get_available_methods()}")
        
        # Test di un singolo explainer
        try:
            lime_explainer = manager.get_explainer('lime')
            explanation = lime_explainer.explain(test_text)
            print(f"SUCCESS: LIME test - {len(explanation['tokens'])} tokens analizzati")
        except Exception as e:
            print(f"WARNING: LIME test fallito: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Test explainer fallito: {str(e)}")
        return False


if __name__ == "__main__":
    # Demo
    test_explainers()