import os
import random
import argparse
from django.shortcuts import render
import torch
from pytorch_transformers import BertTokenizer
os.sys.path.append(os.getcwd() + '/WebProject')
os.sys.path.append(os.getcwd() + '/WebProject/summarization/src')
from models.model_builder import AbsSummarizer
from PreSummdev.src.text_processing import text_refine
from PreSummdev.src.train_abstractive import test_text_abs

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('unused', type=str, nargs=2)
parser.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
parser.add_argument("-mode", default='custom', type=str, choices=['train', 'validate', 'test', 'test_text', 'custom'])
parser.add_argument("-bert_data_path", default='../bert_data_new/cnndm')
parser.add_argument("-model_path", default='../models/')
parser.add_argument("-result_path", default='../results/cnndm')
parser.add_argument("-temp_dir", default='../temp')

parser.add_argument("-batch_size", default=140, type=int)
parser.add_argument("-test_batch_size", default=200, type=int)
parser.add_argument("-max_ndocs_in_batch", default=6, type=int)

parser.add_argument("-max_pos", default=512, type=int)
parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("-load_from_extractive", default='', type=str)

parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("-lr_bert", default=2e-3, type=float)
parser.add_argument("-lr_dec", default=2e-3, type=float)
parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-dec_dropout", default=0.2, type=float)
parser.add_argument("-dec_layers", default=6, type=int)
parser.add_argument("-dec_hidden_size", default=768, type=int)
parser.add_argument("-dec_heads", default=8, type=int)
parser.add_argument("-dec_ff_size", default=2048, type=int)
parser.add_argument("-enc_hidden_size", default=512, type=int)
parser.add_argument("-enc_ff_size", default=512, type=int)
parser.add_argument("-enc_dropout", default=0.2, type=float)
parser.add_argument("-enc_layers", default=6, type=int)

# params for EXT
parser.add_argument("-ext_dropout", default=0.2, type=float)
parser.add_argument("-ext_layers", default=2, type=int)
parser.add_argument("-ext_hidden_size", default=768, type=int)
parser.add_argument("-ext_heads", default=8, type=int)
parser.add_argument("-ext_ff_size", default=2048, type=int)

parser.add_argument("-label_smoothing", default=0.1, type=float)
parser.add_argument("-generator_shard_size", default=32, type=int)
parser.add_argument("-alpha",  default=0.6, type=float)
parser.add_argument("-beam_size", default=5, type=int)
parser.add_argument("-min_length", default=15, type=int)
parser.add_argument("-max_length", default=150, type=int)
parser.add_argument("-max_tgt_len", default=140, type=int)



parser.add_argument("-param_init", default=0, type=float)
parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-optim", default='adam', type=str)
parser.add_argument("-lr", default=1, type=float)
parser.add_argument("-beta1", default= 0.9, type=float)
parser.add_argument("-beta2", default=0.999, type=float)
parser.add_argument("-warmup_steps", default=8000, type=int)
parser.add_argument("-warmup_steps_bert", default=8000, type=int)
parser.add_argument("-warmup_steps_dec", default=8000, type=int)
parser.add_argument("-max_grad_norm", default=0, type=float)

parser.add_argument("-save_checkpoint_steps", default=5, type=int)
parser.add_argument("-accum_count", default=1, type=int)
parser.add_argument("-report_every", default=1, type=int)
parser.add_argument("-train_steps", default=1000, type=int)
parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)


parser.add_argument('-visible_gpus', default='-1', type=str)
parser.add_argument('-gpu_ranks', default='0', type=str)
parser.add_argument('-log_file', default='../logs/cnndm.log')
parser.add_argument('-seed', default=666, type=int)

parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("-test_from", default='./Webproject/summarization/models/model_step_148000.pt')
parser.add_argument("-test_start_from", default=-1, type=int)

parser.add_argument("-train_from", default='')
parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

args = parser.parse_args()
args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
args.world_size = len(args.gpu_ranks)
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

device = "cpu" if args.visible_gpus == '-1' else "cuda"
device_id = 0 if device == "cuda" else -1

checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
opt = vars(checkpoint['opt'])

model = AbsSummarizer(args, device, checkpoint)
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)

def getrespond(request):
    context = {}
    if(request.POST):
        print(request.POST["Language"])
        #print(request.POST)
        context["input"] = request.POST["input_block"]
        if (request.POST["Language"] != "English"):
            context["err"] = "ERROR: The language is not supported currently. Please try later."
            return render(request, "errorpage.html", context)
        data = context["input"]
        if (len(data) < 40):
            context["err"] = "Please Type in a longer text. This is not a MT model."
            return render(request, "errorpage.html", context)
        r = random.randint(0, 10000)
        out_file = './Webproject/summarization/raw_data/raw_src' + str(r) +'.txt'
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(data)
        #print(data)
        text_src = out_file
        final_path = './Webproject/summarization/results/final_opt' + str(r) +'.txt'
        #print(text_src)
        #print(final_path)
        try:
            test_text_abs(args, model, tokenizer, text_src, final_path)
            data = text_refine(text_src, final_path + '.-1.candidate', final_path)
        except:
            context["err"] = "An error occurs when the model executes. Please retry."
            return render(request, "errorpage.html", context)
        #print(ret)
        #if(ret != 0):
        #    context["err"] = "ERROR: the model terminnated with none-zero exit code."
        #    return render(request, 'errorpage.html', context)
        #result_file = './PreSumm-dev/results/final_opt' + str(r) +'.txt'
        #with open(result_file, 'r', encoding='utf-8') as f:
        #    data = f.read()
        context["ret"] = data
        return render(request, 'mainpage.html', context)