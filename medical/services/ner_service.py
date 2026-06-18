from transformers import pipeline
import torch
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NERService:

    def __init__(self):

        self.pipe = pipeline("token-classification", 
                           model="Clinical-AI-Apollo/Medical-NER", 
                           aggregation_strategy='simple',
                           device=0 if torch.cuda.is_available() else -1)
  
    def process(self, text, options, term_types):

        result = self.pipe(text)
        

        if isinstance(result, dict):
            result = result.get('entities', [])
        

        combined_result = self._combine_entities(result, text, options)
        

        non_overlapping_result = self._remove_overlapping_entities(combined_result)
        

        filtered_result = self._filter_entities(non_overlapping_result, term_types)
        
        return {
            "text": text,
            "entities": filtered_result
        }

    def _combine_entities(self, result, text, options):

        combined_result = []
        i = 0
        while i < len(result):
            entity = result[i]
            entity['score'] = float(entity['score'])

            if options['combineBioStructure'] and entity['entity_group'] in ['SIGN_SYMPTOM', 'DISEASE_DISORDER']:
                combined_entity = self._try_combine_with_bio_structure(result, i, text)
                if combined_entity:
                    combined_result.append(combined_entity)
                    i += 1
                    continue
            combined_result.append(entity)
            i += 1
        return combined_result

    def _try_combine_with_bio_structure(self, result, i, text):

        if i > 0 and result[i-1]['entity_group'] == 'BIOLOGICAL_STRUCTURE':
            return self._create_combined_entity(result[i-1], result[i], text)
        elif i < len(result) - 1 and result[i+1]['entity_group'] == 'BIOLOGICAL_STRUCTURE':
            return self._create_combined_entity(result[i], result[i+1], text)
        return None

    def _create_combined_entity(self, entity1, entity2, text):

        start = min(entity1['start'], entity2['start'])
        end = max(entity1['end'], entity2['end'])
        word = text[start:end]
        return {
            'entity_group': 'COMBINED_BIO_SYMPTOM',
            'word': word,
            'start': start,
            'end': end,
            'score': (entity1['score'] + entity2['score']) / 2,
            'original_entities': [entity1, entity2]
        }

    def _remove_overlapping_entities(self, entities):

        sorted_entities = sorted(entities, key=lambda x: (x['start'], -x['end'], -x['score']))
        non_overlapping = []
        last_end = -1

        i = 0
        while i < len(sorted_entities):
            current = sorted_entities[i]
            
            if current['start'] >= last_end:
                non_overlapping.append(current)
                last_end = current['end']
                i += 1
            else:
                same_span = [current]
                j = i + 1
                while j < len(sorted_entities) and sorted_entities[j]['start'] == current['start'] and sorted_entities[j]['end'] == current['end']:
                    same_span.append(sorted_entities[j])
                    j += 1
                
                best_entity = max(same_span, key=lambda x: x['score'])
                if best_entity['end'] > last_end:
                    non_overlapping.append(best_entity)
                    last_end = best_entity['end']
                
                i = j

        return non_overlapping

    def _filter_entities(self, entities, term_types):

        filtered_result = []
        for entity in entities:
            if term_types.get('allMedicalTerms', False):
                filtered_result.append(entity)
            elif (term_types.get('symptom', False) and entity['entity_group'] in ['SIGN_SYMPTOM', 'COMBINED_BIO_SYMPTOM']) or \
                 (term_types.get('disease', False) and entity['entity_group'] == 'DISEASE_DISORDER') or \
                 (term_types.get('therapeuticProcedure', False) and entity['entity_group'] == 'THERAPEUTIC_PROCEDURE'):
                filtered_result.append(entity)
        return filtered_result




