import opencc
from typing import Sequence
from transformers import pipeline
from ievals.modules.qa_evaluators.hf_chat import HF_Chat_Evaluator


class HF_Evaluator(HF_Chat_Evaluator):

    def __init__(
        self, choices, k, model_name, switch_zh_hans=False
    ):
        super().__init__(choices, model_name, k)

        self.converter = None
        if switch_zh_hans:
            self.converter = opencc.OpenCC("t2s.json")

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     model_name, local_files_only=True
        # )
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     local_files_only=True
        # ).eval()

        # self.model.generation_config.do_sample = False
        # self.model.generation_config.repetition_penalty = 1.0

        self.pipeline = pipeline(
            model=model_name,
            device_map="auto",
        )

    def eval_subject(
        self,
        subject_name,
        test_df,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        cot=False,
    ):

        correct_num = 0
        if save_result_dir:
            result = []
            score = []

        q_history = None
        if few_shot:
            history = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            history = []
        answers = list(test_df["answer"])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot)
            response = None
            timeout_counter = 0
            text = ""

            if self.converter:
                question = self.converter.convert(question)
                history = [self.converter.convert(hist) for hist in history]
            # better to check for history accuracy
            while response is None and timeout_counter <= 30:
                try:
                    # response, _ = self.model.chat(
                    #     self.tokenizer, question, history=history
                    # )
                    response = self.pipeline(
                        question, do_sample = False, repetition_penalty = 1.0
                    )
                    if isinstance(response, Sequence):
                        generated_text = response[0]
                        response = generated_text.get('generated_text', None)
                    else:
                        response = response.get('generated_text', None)

                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter += 1
                    print(msg)
                    sleep(5)
                    continue

            if response == None:
                response_str = ""
            else:
                response_str = response
            if cot:  # simplified chinese
                ans_list = re.findall(r"答案是(.+?)。", response_str)
                if self.converter:  # simplified chinese
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"答案为(.+?)", response_str)
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"选项(.+?)是正确的", response_str)
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"因此，选项(.+?)", response_str)
                else:
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"答案為(.+?)", response_str)
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"選項(.+?)是正確的", response_str)
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"因此，選項(.+?)", response_str)

                if len(ans_list) == 0:
                    correct = 0
                else:
                    if self.exact_match(ans_list[-1], row["answer"]):
                        correct_num += 1
                        correct = 1
                    else:
                        correct = 0
            else:
                response_str = response_str.strip()
                if len(response_str) > 0:
                    ans_list = self.extract_ans(response_str)
                    if len(ans_list) > 0 and (ans_list[-1] == row["answer"]):
                        correct_num += 1
                        correct = 1
                    else:
                        correct = 0
                else:
                    correct = 0
            if save_result_dir:
                result.append(response_str)
                score.append(correct)
        correct_ratio = 100 * correct_num / len(answers)

        if save_result_dir:
            test_df["model_output"] = result
            test_df["correctness"] = score
            test_df.to_csv(
                os.path.join(save_result_dir, f"{subject_name}_val.csv"),
                encoding="utf-8",
                index=False,
            )
        return correct_ratio