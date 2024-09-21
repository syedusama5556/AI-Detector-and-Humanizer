class Gramformer:

  def __init__(self, grammar_tokenizer, grammar_model, use_gpu=False):
    import errant
    import en_core_web_sm

    # Load spaCy model and Errant for grammatical edits
    nlp = en_core_web_sm.load()
    self.annotator = errant.load('en', nlp)
    
    # Use GPU if available
    if use_gpu:
      self.device = "cuda:0"
    else:
      self.device = "cpu"

    # Assign the pre-loaded model and tokenizer
    self.correction_tokenizer = grammar_tokenizer
    self.correction_model = grammar_model.to(self.device)
    self.model_loaded = True
    print("[Gramformer] Pre-loaded grammar model initialized.")

  def correct(self, input_sentence, max_candidates=1):
    if self.model_loaded:
        # Add prefix required for the grammar model
        correction_prefix = "gec: "
        input_sentence = correction_prefix + input_sentence
        input_ids = self.correction_tokenizer.encode(input_sentence, return_tensors='pt')
        input_ids = input_ids.to(self.device)

        # Generate corrections
        preds = self.correction_model.generate(
            input_ids,
            do_sample=True,
            max_length=512,  # Ensure this matches the maximum length for your model
            num_beams=7,
            early_stopping=True,
            num_return_sequences=max_candidates
        )

        corrected = []
        for pred in preds:
            corrected.append(self.correction_tokenizer.decode(pred, skip_special_tokens=True).strip())  # Use append instead of add

        return corrected
    else:
        print("Model is not loaded.")
        return None

  def highlight(self, orig, cor):
    # This method highlights the edits between the original and corrected text
    edits = self._get_edits(orig, cor)
    orig_tokens = orig.split()
    ignore_indexes = []

    for edit in edits:
      edit_type = edit[0]
      edit_str_start = edit[1]
      edit_spos = edit[2]
      edit_epos = edit[3]
      edit_str_end = edit[4]

      # Handle cases where tokens are ignored or highlighted
      for i in range(edit_spos+1, edit_epos):
        ignore_indexes.append(i)

      if edit_str_start == "":
        new_edit_str = orig_tokens[edit_spos - 1] if edit_spos - 1 >= 0 else orig_tokens[edit_spos + 1]
        if edit_type == "PUNCT":
          st = f"<a type='{edit_type}' edit='{edit_str_end}'>{new_edit_str}</a>"
        else:
          st = f"<a type='{edit_type}' edit='{new_edit_str} {edit_str_end}'>{new_edit_str}</a>"
        orig_tokens[edit_spos] = st
      elif edit_str_end == "":
        st = f"<d type='{edit_type}' edit=''>{edit_str_start}</d>"
        orig_tokens[edit_spos] = st
      else:
        st = f"<c type='{edit_type}' edit='{edit_str_end}'>{edit_str_start}</c>"
        orig_tokens[edit_spos] = st

    for i in sorted(ignore_indexes, reverse=True):
      del(orig_tokens[i])

    return " ".join(orig_tokens)

  def detect(self, input_sentence):
    # Method to detect errors (to be implemented if needed)
    pass

  def _get_edits(self, orig, cor):
    orig = self.annotator.parse(orig)
    cor = self.annotator.parse(cor)
    alignment = self.annotator.align(orig, cor)
    edits = self.annotator.merge(alignment)

    if len(edits) == 0:
      return []

    edit_annotations = []
    for e in edits:
      e = self.annotator.classify(e)
      edit_annotations.append((e.type[2:], e.o_str, e.o_start, e.o_end, e.c_str, e.c_start, e.c_end))
    
    return edit_annotations

  def get_edits(self, orig, cor):
    return self._get_edits(orig, cor)
