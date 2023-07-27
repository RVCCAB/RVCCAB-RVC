import os
import shutil
import sys
import json  # Mangio fork using json for preset saving
import math
import zipfile
import signal

now_dir = os.getcwd()
sys.path.append(now_dir)
import traceback, pdb
import warnings
from pydub import AudioSegment
import os, shutil, wave
import numpy as np
import torch
import re
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import logging
import threading
from random import shuffle
from subprocess import Popen
from time import sleep
import subprocess
import faiss
import ffmpeg
import gradio as gr
import soundfile as sf
from config import Config
from fairseq import checkpoint_utils
from i18n import I18nAuto
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
from infer_uvr5 import _audio_pre_, _audio_pre_new
from MDXNet import MDXNetDereverb
from my_utils import load_audio
from train.process_ckpt import change_info, extract_small_model, merge, show_info
from vc_infer_pipeline import VC
from sklearn.cluster import MiniBatchKMeans
import random
import string
logging.getLogger("numba").setLevel(logging.WARNING)


tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

DoFormant = False
Quefrency = 8.0
Timbre = 1.2

with open('formanting.txt', 'w+') as fsf:
    fsf.truncate(0)

    fsf.writelines([str(DoFormant) + '\n', str(Quefrency) + '\n', str(Timbre) + '\n'])
    

config = Config()
i18n = I18nAuto()
i18n.print()
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

isinterrupted = 0

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


hubert_model = None


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "./logs/"
global audio_root
audio_root = "audios"
global input_audio_path0
global input_audio_path1                                        
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []

global indexes_list
indexes_list=[]

audio_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s\\%s" % (root, name))
            
for root, dirs, files in os.walk(audio_root, topdown=False):
    for name in files:
        if name.endswith(".wav") and ".mp3" and ".m4a":
            audio_paths.append("%s/%s" % (root, name))
            
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

def check_for_name():
    if len(names) > 0:
        return sorted(names)[0]
    else:
        return ''

def get_index():
    if check_for_name() != '':
        chosen_model=sorted(names)[0].split(".")[0]
        logs_path="./logs/"+chosen_model
        if os.path.exists(logs_path):
            for file in os.listdir(logs_path):
                if file.endswith(".index"):
                    return os.path.join(logs_path, file).replace('\\','/')
            return ''
        else:
            return ''

def get_indexes():
    for dirpath, dirnames, filenames in os.walk("./logs/"):
        for filename in filenames:
            if filename.endswith(".index") and "trained" not in filename:
                indexes_list.append(os.path.join(dirpath,filename).replace('\\','/'))
    if len(indexes_list) > 0:
        return indexes_list
    else:
        return ''

fshift_presets_list = []

def get_fshift_presets():
    fshift_presets_list = []
    for dirpath, dirnames, filenames in os.walk("./formantshiftcfg/"):
        for filename in filenames:
            if filename.endswith(".txt"):
                fshift_presets_list.append(os.path.join(dirpath,filename).replace('\\','/'))
                
    if len(fshift_presets_list) > 0:
        return fshift_presets_list
    else:
        return ''


def get_audios():
    if check_for_name() != '':
        audios_path= '"' + os.path.abspath(os.getcwd()) + '/audios/'
        if os.path.exists(audios_path):
            for file in os.listdir(audios_path):
                print(audios_path.join(file) + '"')
                return os.path.join(audios_path, file + '"')
            return ''
        else:
            return ''


def vc_single(
    sid,
    input_audio_path0,
    input_audio_path1,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path0 is None or input_audio_path0 is None:
        return "Bir ses dosyası yüklemelisin.", None
    f0_up_key = int(f0_up_key)
    try:
        if input_audio_path0 == '':
            audio = load_audio(input_audio_path1, 16000, DoFormant, Quefrency, Timbre)
            
        else:
            audio = load_audio(input_audio_path0, 16000, DoFormant, Quefrency, Timbre)
            
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("öğrendi", "eklendi")
            )
            if file_index != ""
            else file_index2
        )  # 防止小白写错，自动帮他替换掉
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path1,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0_file=f0_file,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        index_info = (
            "Bu index kullanılıyor: %s." % file_index
            if os.path.exists(file_index)
            else "Index kullanılmadı."
        )
        return "İşlem başarıyla tamamlandı.", (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def vc_multi(
    sid,
    dir_path,
    opt_root,
    paths,
    f0_up_key,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    format1,
    crepe_hop_length,
):
    try:
        dir_path = (
            dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        os.makedirs(opt_root, exist_ok=True)
        try:
            if dir_path != "":
                paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            else:
                paths = [path.name for path in paths]
        except:
            traceback.print_exc()
            paths = [path.name for path in paths]
        infos = []
        for path in paths:
            info, opt = vc_single(
                sid,
                path,
                f0_up_key,
                None,
                f0_method,
                file_index,
                file_index2,
                # file_big_npy,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                crepe_hop_length
            )
            if "Success" in info:
                try:
                    tgt_sr, audio_opt = opt
                    if format1 in ["wav", "flac", "mp3", "ogg", "aac"]:
                        sf.write(
                            "%s/%s.%s" % (opt_root, os.path.basename(path), format1),
                            audio_opt,
                            tgt_sr,
                        )
                    else:
                        path = "%s/%s.wav" % (opt_root, os.path.basename(path))
                        sf.write(
                            path,
                            audio_opt,
                            tgt_sr,
                        )
                        if os.path.exists(path):
                            os.system(
                                "ffmpeg -i %s -vn %s -q:a 2 -y"
                                % (path, path[:-4] + ".%s" % format1)
                            )
                except:
                    info += traceback.format_exc()
            infos.append("%s->%s" % (os.path.basename(path), info))
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        else:
            func = _audio_pre_ if "DeEcho" not in model_name else _audio_pre_new
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=config.device,
                is_half=config.is_half,
            )
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (tmp, os.path.basename(inp_path))
                os.system(
                    "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    yield "\n".join(infos)


# 一个选项卡全局只能有一个音色
def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###楼下不这么折腾清理不干净
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return ({"visible": False, "__type__": "update"}, {"visible": False, "__type__": "update"}, {"visible": False, "__type__": "update"})
    person = "%s/%s" % (weight_root, sid)
    print("yükleniyor: %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 0:
        to_return_protect0 = to_return_protect1 = {
            "visible": False,
            "value": 0.5,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": True,
            "value": to_return_protect1,
            "__type__": "update",
        }
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return (
        {"visible": False, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )


def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    audio_paths = []
    audios_path=os.path.abspath(os.getcwd()) + "/audios/"
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    for file in os.listdir(audios_path):
                if file.endswith((".wav",".mp3",".m4a")):
                    audio_paths.append("%s/%s" % (audio_root, file))
    return {"choices": sorted(names), "__type__": "update"}, {"choices": sorted(index_paths), "__type__": "update"}, {"choices": sorted(audio_paths), "__type__": "update"}

def change_choices2():
    audio_files=[]
    for filename in os.listdir("./audios"):
        if filename.endswith(('.wav','.mp3')):
            audio_files.append("%s/%s" % (audio_root, filename))
    return {"choices": sorted(audio_files), "__type__": "update"}

def clean():
    return ({"value": "", "__type__": "update"})
    

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True

def formant_enabled(cbox, qfrency, tmbre, frmntapply, formantpreset, formant_refresh_button):
    
    if (cbox):

        DoFormant = True
        with open('formanting.txt', 'w') as fxxf:
            fxxf.truncate(0)

            fxxf.writelines([str(DoFormant) + '\n', str(Quefrency) + '\n', str(Timbre) + '\n'])
        #print(f"is checked? - {cbox}\ngot {DoFormant}")
        
        return (
            {"value": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
        )
        
        
    else:
        
        DoFormant = False
        with open('formanting.txt', 'w') as fxf:
            fxf.truncate(0)

            fxf.writelines([str(DoFormant) + '\n', str(Quefrency) + '\n', str(Timbre) + '\n'])
        #print(f"is checked? - {cbox}\ngot {DoFormant}")
        return (
            {"value": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
        )
        

def formant_apply(qfrency, tmbre):
    Quefrency = qfrency
    Timbre = tmbre
    DoFormant = True
    
    with open('formanting.txt', 'w') as fxxxf:
        fxxxf.truncate(0)

        fxxxf.writelines([str(DoFormant) + '\n', str(Quefrency) + '\n', str(Timbre) + '\n'])
    return ({"value": Quefrency, "__type__": "update"}, {"value": Timbre, "__type__": "update"})

def update_fshift_presets(preset, qfrency, tmbre):
    
    qfrency, tmbre = preset_apply(preset, qfrency, tmbre)
    
    if (str(preset) != ''):
        with open(str(preset), 'r') as p:
            content = p.readlines()
            qfrency, tmbre = content[0].split('\n')[0], content[1]
            
            formant_apply(qfrency, tmbre)
    else:
        pass
    return (
        {"choices": get_fshift_presets(), "__type__": "update"},
        {"value": qfrency, "__type__": "update"},
        {"value": tmbre, "__type__": "update"},
    )


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    cmd = (
        config.python_cmd
        + " trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s "
        % (trainset_dir, sr, n_p, now_dir, exp_dir)
        + str(config.noparallel)
    )
    print(cmd)
    p = Popen(cmd, shell=True)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    print(log)
    yield log


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, echl):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        cmd = config.python_cmd + " extract_f0_print.py %s/logs/%s %s %s %s" % (
            now_dir,
            exp_dir,
            n_p,
            f0method,
            echl,
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
        ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
        done = [False]
        threading.Thread(
            target=if_done,
            args=(
                done,
                p,
            ),
        ).start()
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        print(log)
        yield log
    ####对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            config.python_cmd
            + " extract_feature_print.py %s %s %s %s %s/logs/%s %s"
            % (
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
            )
        )
        print(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    print(log)
    yield log


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    if_pretrained_generator_exist = os.access(
        "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        print(
            "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2),
            "ön eğitilmiş model bulunamadı.",
        )
    if not if_pretrained_discriminator_exist:
        print(
            "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2),
            "ön eğitilmiş model bulunamadı.",
        )
    return (
        "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_generator_exist
        else "",
        "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_discriminator_exist
        else "",
    )


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    if_pretrained_generator_exist = os.access(
        "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        print(
            "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2),
            "not exist, will not use pretrained model",
        )
    if not if_pretrained_discriminator_exist:
        print(
            "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2),
            "not exist, will not use pretrained model",
        )
    return (
        "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_generator_exist
        else "",
        "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_discriminator_exist
        else "",
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19, step2b, gpus6, gpu_info9, extraction_crepe_hop_length, but2, info2):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    if_pretrained_generator_exist = os.access(
        "pretrained%s/f0G%s.pth" % (path_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "pretrained%s/f0D%s.pth" % (path_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        print(
            "pretrained%s/f0G%s.pth" % (path_str, sr2),
            "not exist, will not use pretrained model",
        )
    if not if_pretrained_discriminator_exist:
        print(
            "pretrained%s/f0D%s.pth" % (path_str, sr2),
            "not exist, will not use pretrained model",
        )
    
    if if_f0_3:
        return (
            {"visible": True, "__type__": "update"},
            "pretrained%s/f0G%s.pth" % (path_str, sr2)
            if if_pretrained_generator_exist
            else "",
            "pretrained%s/f0D%s.pth" % (path_str, sr2)
            if if_pretrained_discriminator_exist
            else "",
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
        )
        
    return (
        {"visible": False, "__type__": "update"},
        ("pretrained%s/G%s.pth" % (path_str, sr2))
        if if_pretrained_generator_exist
        else "",
        ("pretrained%s/D%s.pth" % (path_str, sr2))
        if if_pretrained_discriminator_exist
        else "",
        {"visible": False, "__type__": "update"},
        {"visible": False, "__type__": "update"},
        {"visible": False, "__type__": "update"},
        {"visible": False, "__type__": "update"},
        {"visible": False, "__type__": "update"},
        {"visible": False, "__type__": "update"},
    )


global log_interval


def set_log_interval(exp_dir, batch_size12):
    log_interval = 1

    folder_path = os.path.join(exp_dir, "1_16k_wavs")

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        if wav_files:
            sample_size = len(wav_files)
            log_interval = math.ceil(sample_size / batch_size12)
            if log_interval > 1:
                log_interval += 1

    return log_interval


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )

    log_interval = set_log_interval(exp_dir, batch_size12)

    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    print("use gpus:", gpus16)
    if pretrained_G14 == "":
        print("no pretrained Generator")
    if pretrained_D15 == "":
        print("no pretrained Discriminator")
    if gpus16:
        ####
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s -li %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == True else 0,
                1 if if_cache_gpu17 == True else 0,
                1 if if_save_every_weights18 == True else 0,
                version19,
                log_interval,
            )
        )
    else:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s -li %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "\b",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "\b",
                1 if if_save_latest13 == True else 0,
                1 if if_cache_gpu17 == True else 0,
                1 if if_save_every_weights18 == True else 0,
                version19,
                log_interval,
            )
        )
    print(cmd)
    global p
    p = Popen(cmd, shell=True, cwd=now_dir)
    global PID
    PID = p.pid

    p.wait()
    return ("Öğretme işlemi tamamlandı. Kayıtlar train.log dosyasına kaydedildi.", {"visible": False, "__type__": "update"}, {"visible": True, "__type__": "update"})


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        # if(1):
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            print(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("öğreniyor...")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    # faiss.write_index(index, '%s/trained_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    infos.append("ekleniyor...")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "Index öğrenimi bitti. added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)

#def setBoolean(status): #true to false and vice versa / not implemented yet, dont touch!!!!!!!
#    status = not status
#    return status
    
# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    echl
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    model_log_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    preprocess_log_path = "%s/preprocess.log" % model_log_dir
    extract_f0_feature_log_path = "%s/extract_f0_feature.log" % model_log_dir
    gt_wavs_dir = "%s/0_gt_wavs" % model_log_dir
    feature_dir = (
        "%s/3_feature256" % model_log_dir
        if version19 == "v1"
        else "%s/3_feature768" % model_log_dir
    )

    os.makedirs(model_log_dir, exist_ok=True)
    #########step1:处理数据
    open(preprocess_log_path, "w").close()
    cmd = (
        config.python_cmd
        + " trainset_preprocess_pipeline_print.py %s %s %s %s "
        % (trainset_dir4, sr_dict[sr2], np7, model_log_dir)
        + str(config.noparallel)
    )
    yield get_info_str(i18n("step1:正在处理数据"))
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True)
    p.wait()
    with open(preprocess_log_path, "r") as f:
        print(f.read())
    #########step2a:提取音高
    open(extract_f0_feature_log_path, "w")
    if if_f0_3:
        yield get_info_str("step2a:正在提取音高")
        cmd = config.python_cmd + " extract_f0_print.py %s %s %s %s" % (
            model_log_dir,
            np7,
            f0method8,
            echl
        )
        yield get_info_str(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        p.wait()
        with open(extract_f0_feature_log_path, "r") as f:
            print(f.read())
    else:
        yield get_info_str(i18n("step2a:无需提取音高"))
    #######step2b:提取特征
    yield get_info_str(i18n("step2b:正在提取特征"))
    gpus = gpus16.split("-")
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = config.python_cmd + " extract_feature_print.py %s %s %s %s %s %s" % (
            config.device,
            leng,
            idx,
            n_g,
            model_log_dir,
            version19,
        )
        yield get_info_str(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    for p in ps:
        p.wait()
    with open(extract_f0_feature_log_path, "r") as f:
        print(f.read())
    #######step3a:训练模型
    yield get_info_str(i18n("step3a:正在训练模型"))
    # 生成filelist
    if if_f0_3:
        f0_dir = "%s/2a_f0" % model_log_dir
        f0nsf_dir = "%s/2b-f0nsf" % model_log_dir
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % model_log_dir, "w") as f:
        f.write("\n".join(opt))
    yield get_info_str("write filelist done")
    if gpus16:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == True else 0,
                1 if if_cache_gpu17 == True else 0,
                1 if if_save_every_weights18 == True else 0,
                version19,
            )
        )
    else:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == True else 0,
                1 if if_cache_gpu17 == True else 0,
                1 if if_save_every_weights18 == True else 0,
                version19,
            )
        )
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    yield get_info_str(i18n("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"))
    #######step3b:训练索引
    npys = []
    listdir_res = list(os.listdir(feature_dir))
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5:
        # if(1):
        info = "Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0]
        print(info)
        yield get_info_str(info)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            print(info)
            yield get_info_str(info)

    np.save("%s/total_fea.npy" % model_log_dir, big_npy)

    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    yield get_info_str("%s,%s" % (big_npy.shape, n_ivf))
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    yield get_info_str("training index")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    yield get_info_str("adding index")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    yield get_info_str(
        "成功构建索引, added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    yield get_info_str(i18n("全流程结束！"))


#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


def export_onnx(ModelPath, ExportedPath):
    cpt = torch.load(ModelPath, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768

    test_phone = torch.rand(1, 200, vec_channels)  # hidden unit
    test_phone_lengths = torch.tensor([200]).long()  # hidden unit 长度（貌似没啥用）
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # 基频（单位赫兹）
    test_pitchf = torch.rand(1, 200)  # nsf基频
    test_ds = torch.LongTensor([0])  # 说话人ID
    test_rnd = torch.rand(1, 192, 200)  # 噪声（加入随机因子）

    device = "cpu"  # 导出时设备（不影响使用模型）


    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], is_half=False, version=cpt.get("version", "v1")
    )  # fp32导出（C++要支持fp16必须手动将内存重新排列所以暂时不用fp16）
    net_g.load_state_dict(cpt["weight"], strict=False)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = [
        "audio",
    ]
    # net_g.construct_spkmixmap(n_speaker) 多角色混合轨道导出
    torch.onnx.export(
        net_g,
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
            test_rnd.to(device),
        ),
        ExportedPath,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=13,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    return "Finished"


#region Mangio-RVC-Fork CLI App
import re as regex
import scipy.io.wavfile as wavfile

cli_current_page = "HOME"

def cli_split_command(com):
    exp = r'(?:(?<=\s)|^)"(.*?)"(?=\s|$)|(\S+)'
    split_array = regex.findall(exp, com)
    split_array = [group[0] if group[0] else group[1] for group in split_array]
    return split_array

def execute_generator_function(genObject):
    for _ in genObject: pass

def cli_infer(com):
    # get VC first
    com = cli_split_command(com)
    model_name = com[0]
    source_audio_path = com[1]
    output_file_name = com[2]
    feature_index_path = com[3]
    f0_file = None # Not Implemented Yet

    # Get parameters for inference
    speaker_id = int(com[4])
    transposition = float(com[5])
    f0_method = com[6]
    crepe_hop_length = int(com[7])
    harvest_median_filter = int(com[8])
    resample = int(com[9])
    mix = float(com[10])
    feature_ratio = float(com[11])
    protection_amnt = float(com[12])
    #####
    
    print("Mangio-RVC-Fork Infer-CLI: Starting the inference...")
    vc_data = get_vc(model_name)
    print(vc_data)
    print("Mangio-RVC-Fork Infer-CLI: Performing inference...")
    conversion_data = vc_single(
        speaker_id,
        source_audio_path,
        source_audio_path,
        transposition,
        f0_file,
        f0_method,
        feature_index_path,
        feature_index_path,
        feature_ratio,
        harvest_median_filter,
        resample,
        mix,
        protection_amnt,
        crepe_hop_length,        
    )
    if "Success." in conversion_data[0]:
        print("Mangio-RVC-Fork Infer-CLI: Inference succeeded. Writing to %s/%s..." % ('audio-outputs', output_file_name))
        wavfile.write('%s/%s' % ('audio-outputs', output_file_name), conversion_data[1][0], conversion_data[1][1])
        print("Mangio-RVC-Fork Infer-CLI: Finished! Saved output to %s/%s" % ('audio-outputs', output_file_name))
    else:
        print("Mangio-RVC-Fork Infer-CLI: Inference failed. Here's the traceback: ")
        print(conversion_data[0])

def cli_pre_process(com):
    com = cli_split_command(com)
    model_name = com[0]
    trainset_directory = com[1]
    sample_rate = com[2]
    num_processes = int(com[3])

    print("Mangio-RVC-Fork Pre-process: Starting...")
    generator = preprocess_dataset(
        trainset_directory, 
        model_name, 
        sample_rate, 
        num_processes
    )
    execute_generator_function(generator)
    print("Mangio-RVC-Fork Pre-process: Finished")

def cli_extract_feature(com):
    com = cli_split_command(com)
    model_name = com[0]
    gpus = com[1]
    num_processes = int(com[2])
    has_pitch_guidance = True if (int(com[3]) == 1) else False
    f0_method = com[4]
    crepe_hop_length = int(com[5])
    version = com[6] # v1 or v2
    
    print("Mangio-RVC-CLI: Extract Feature Has Pitch: " + str(has_pitch_guidance))
    print("Mangio-RVC-CLI: Extract Feature Version: " + str(version))
    print("Mangio-RVC-Fork Feature Extraction: Starting...")
    generator = extract_f0_feature(
        gpus, 
        num_processes, 
        f0_method, 
        has_pitch_guidance, 
        model_name, 
        version, 
        crepe_hop_length
    )
    execute_generator_function(generator)
    print("Mangio-RVC-Fork Feature Extraction: Finished")

def cli_train(com):
    com = cli_split_command(com)
    model_name = com[0]
    sample_rate = com[1]
    has_pitch_guidance = True if (int(com[2]) == 1) else False
    speaker_id = int(com[3])
    save_epoch_iteration = int(com[4])
    total_epoch = int(com[5]) # 10000
    batch_size = int(com[6])
    gpu_card_slot_numbers = com[7]
    if_save_latest = True if (int(com[8]) == 1) else False
    if_cache_gpu = True if (int(com[9]) == 1) else False
    if_save_every_weight = True if (int(com[10]) == 1) else False
    version = com[11]

    pretrained_base = "pretrained/" if version == "v1" else "pretrained_v2/" 
    
    g_pretrained_path = "%sf0G%s.pth" % (pretrained_base, sample_rate)
    d_pretrained_path = "%sf0D%s.pth" % (pretrained_base, sample_rate)

    print("Mangio-RVC-Fork Train-CLI: Training...")
    click_train(
        model_name,
        sample_rate,
        has_pitch_guidance,
        speaker_id,
        save_epoch_iteration,
        total_epoch,
        batch_size,
        if_save_latest,
        g_pretrained_path,
        d_pretrained_path,
        gpu_card_slot_numbers,
        if_cache_gpu,
        if_save_every_weight,
        version
    )

def cli_train_feature(com):
    com = cli_split_command(com)
    model_name = com[0]
    version = com[1]
    print("Mangio-RVC-Fork Train Feature Index-CLI: Training... Please wait")
    generator = train_index(
        model_name,
        version
    )
    execute_generator_function(generator)
    print("Mangio-RVC-Fork Train Feature Index-CLI: Done!")

def cli_extract_model(com):
    com = cli_split_command(com)
    model_path = com[0]
    save_name = com[1]
    sample_rate = com[2]
    has_pitch_guidance = com[3]
    info = com[4]
    version = com[5]
    extract_small_model_process = extract_small_model(
        model_path,
        save_name,
        sample_rate,
        has_pitch_guidance,
        info,
        version
    )
    if extract_small_model_process == "Success.":
        print("Mangio-RVC-Fork Extract Small Model: Success!")
    else:
        print(str(extract_small_model_process))        
        print("Mangio-RVC-Fork Extract Small Model: Failed!")


def preset_apply(preset, qfer, tmbr):
    if str(preset) != '':
        with open(str(preset), 'r') as p:
            content = p.readlines()
            qfer, tmbr = content[0].split('\n')[0], content[1]
            
            formant_apply(qfer, tmbr)
    else:
        pass
    return ({"value": qfer, "__type__": "update"}, {"value": tmbr, "__type__": "update"})

def print_page_details():
    if cli_current_page == "HOME":
        print("    go home            : Takes you back to home with a navigation list.")
        print("    go infer           : Takes you to inference command execution.\n")
        print("    go pre-process     : Takes you to training step.1) pre-process command execution.")
        print("    go extract-feature : Takes you to training step.2) extract-feature command execution.")
        print("    go train           : Takes you to training step.3) being or continue training command execution.")
        print("    go train-feature   : Takes you to the train feature index command execution.\n")
        print("    go extract-model   : Takes you to the extract small model command execution.")
    elif cli_current_page == "INFER":
        print("    arg 1) model name with .pth in ./weights: mi-test.pth")
        print("    arg 2) source audio path: myFolder\\MySource.wav")
        print("    arg 3) output file name to be placed in './audio-outputs': MyTest.wav")
        print("    arg 4) feature index file path: logs/mi-test/added_IVF3042_Flat_nprobe_1.index")
        print("    arg 5) speaker id: 0")
        print("    arg 6) transposition: 0")
        print("    arg 7) f0 method: harvest (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny)")
        print("    arg 8) crepe hop length: 160")
        print("    arg 9) harvest median filter radius: 3 (0-7)")
        print("    arg 10) post resample rate: 0")
        print("    arg 11) mix volume envelope: 1")
        print("    arg 12) feature index ratio: 0.78 (0-1)")
        print("    arg 13) Voiceless Consonant Protection (Less Artifact): 0.33 (Smaller number = more protection. 0.50 means Dont Use.) \n")
        print("Example: mi-test.pth saudio/Sidney.wav myTest.wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33")
    elif cli_current_page == "PRE-PROCESS":
        print("    arg 1) Model folder name in ./logs: mi-test")
        print("    arg 2) Trainset directory: mydataset (or) E:\\my-data-set")
        print("    arg 3) Sample rate: 40k (32k, 40k, 48k)")
        print("    arg 4) Number of CPU threads to use: 8 \n")
        print("Example: mi-test mydataset 40k 24")
    elif cli_current_page == "EXTRACT-FEATURE":
        print("    arg 1) Model folder name in ./logs: mi-test")
        print("    arg 2) Gpu card slot: 0 (0-1-2 if using 3 GPUs)")
        print("    arg 3) Number of CPU threads to use: 8")
        print("    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)")
        print("    arg 5) f0 Method: harvest (pm, harvest, dio, crepe)")
        print("    arg 6) Crepe hop length: 128")
        print("    arg 7) Version for pre-trained models: v2 (use either v1 or v2)\n")
        print("Example: mi-test 0 24 1 harvest 128 v2")
    elif cli_current_page == "TRAIN":
        print("    arg 1) Model folder name in ./logs: mi-test")
        print("    arg 2) Sample rate: 40k (32k, 40k, 48k)")
        print("    arg 3) Has Pitch Guidance?: 1 (0 for no, 1 for yes)")
        print("    arg 4) speaker id: 0")
        print("    arg 5) Save epoch iteration: 50")
        print("    arg 6) Total epochs: 10000")
        print("    arg 7) Batch size: 8")
        print("    arg 8) Gpu card slot: 0 (0-1-2 if using 3 GPUs)")
        print("    arg 9) Save only the latest checkpoint: 0 (0 for no, 1 for yes)")
        print("    arg 10) Whether to cache training set to vram: 0 (0 for no, 1 for yes)")
        print("    arg 11) Save extracted small model every generation?: 0 (0 for no, 1 for yes)")
        print("    arg 12) Model architecture version: v2 (use either v1 or v2)\n")
        print("Example: mi-test 40k 1 0 50 10000 8 0 0 0 0 v2")
    elif cli_current_page == "TRAIN-FEATURE":
        print("    arg 1) Model folder name in ./logs: mi-test")
        print("    arg 2) Model architecture version: v2 (use either v1 or v2)\n")
        print("Example: mi-test v2")
    elif cli_current_page == "EXTRACT-MODEL":
        print("    arg 1) Model Path: logs/mi-test/G_168000.pth")
        print("    arg 2) Model save name: MyModel")
        print("    arg 3) Sample rate: 40k (32k, 40k, 48k)")
        print("    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)")
        print('    arg 5) Model information: "My Model"')
        print("    arg 6) Model architecture version: v2 (use either v1 or v2)\n")
        print('Example: logs/mi-test/G_168000.pth MyModel 40k 1 "Created by Cole Mangio" v2')
    print("")

def change_page(page):
    global cli_current_page
    cli_current_page = page
    return 0

def execute_command(com):
    if com == "go home":
        return change_page("HOME")
    elif com == "go infer":
        return change_page("INFER")
    elif com == "go pre-process":
        return change_page("PRE-PROCESS")
    elif com == "go extract-feature":
        return change_page("EXTRACT-FEATURE")
    elif com == "go train":
        return change_page("TRAIN")
    elif com == "go train-feature":
        return change_page("TRAIN-FEATURE")
    elif com == "go extract-model":
        return change_page("EXTRACT-MODEL")
    else:
        if com[:3] == "go ":
            print("page '%s' does not exist!" % com[3:])
            return 0
    
    if cli_current_page == "INFER":
        cli_infer(com)
    elif cli_current_page == "PRE-PROCESS":
        cli_pre_process(com)
    elif cli_current_page == "EXTRACT-FEATURE":
        cli_extract_feature(com)
    elif cli_current_page == "TRAIN":
        cli_train(com)
    elif cli_current_page == "TRAIN-FEATURE":
        cli_train_feature(com)
    elif cli_current_page == "EXTRACT-MODEL":
        cli_extract_model(com)

def cli_navigation_loop():
    while True:
        print("You are currently in '%s':" % cli_current_page)
        print_page_details()
        command = input("%s: " % cli_current_page)
        try:
            execute_command(command)
        except:
            print(traceback.format_exc())

if(config.is_cli):
    print("\n\nMangio-RVC-Fork v2 CLI App!\n")
    print("Welcome to the CLI version of RVC. Please read the documentation on https://github.com/Mangio621/Mangio-RVC-Fork (README.MD) to understand how to use this app.\n")
    cli_navigation_loop()

#endregion

#region RVC WebUI App

def get_presets():
    data = None
    with open('../inference-presets.json', 'r') as file:
        data = json.load(file)
    preset_names = []
    for preset in data['presets']:
        preset_names.append(preset['name'])
    
    return preset_names

def stepdisplay(if_save_every_weights):
    return ({"visible": if_save_every_weights, "__type__": "update"})

def match_index(sid0):
    picked = False
    
    folder = sid0.split('.')[0].split('_')[0]
    parent_dir = "./logs/" + folder
    if os.path.exists(parent_dir):
        for filename in os.listdir(parent_dir.replace('\\','/')):
            if filename.endswith(".index"):
                for i in range(len(indexes_list)):
                    if indexes_list[i] == (os.path.join(("./logs/" + folder), filename).replace('\\','/')):
                        print('regular index found')
                        break
                    else:
                        if indexes_list[i] == (os.path.join(("./logs/" + folder.lower()), filename).replace('\\','/')):
                            print('lowered index found')
                            parent_dir = "./logs/" + folder.lower()
                            break

                index_path=os.path.join(parent_dir.replace('\\','/'), filename.replace('\\','/')).replace('\\','/')
                return (index_path, index_path)
                

    else:
        return ('', '')

def choveraudio():
    return ''


def stoptraining(mim): 
    if int(mim) == 1:
        
        with open("stop.txt", "w+") as tostops:

            
            tostops.writelines('stop')
        #p.terminate()
        #p.kill()
        try:
            os.kill(PID, signal.SIGTERM)
        except Exception as e:
            print(f"İşlemini yapamadık. Çünkü: {e}")
            pass
    else:
        pass
    
    return (
        {"visible": False, "__type__": "update"}, 
        {"visible": True, "__type__": "update"},
    )

def file_base_name(file_name):
    if '.' in file_name:
        zipfile_name = os.path.basename(file_name)
        zipfile_name_without_extension = os.path.splitext(zipfile_name)[0]
        return zipfile_name_without_extension
    else:
        return file_name

def whethercrepeornah(radio):
    mango = True if radio == 'mangio-crepe' or radio == 'mangio-crepe-tiny' else False
    
    return ({"visible": mango, "__type__": "update"})

def download_from_url(url, model):
    url = url.strip()
    if url == '':
        return "Bağlantı adresi boş bırakılamaz."
    zip_dirs = ["/content/zips", "/content/unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs("/content/zips", exist_ok=True)
    os.makedirs("/content/unzips", exist_ok=True)
    zipfile = model + '.zip'
    zipfile_path = '/content/zips/' + zipfile
    MODELEPOCH = ''
    
    if "drive.google.com" in url:
        subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
    elif "mega.nz" in url:
        m = Mega()
        m.download_url(url, '/content/zips')
    else:
        subprocess.run(["wget", url, "-O", f"/content/zips/{zipfile}"])
    for filename in os.listdir("/content/zips"):
        if filename.endswith(".zip"):
            zipfile_path = os.path.join("/content/zips/",filename)
            shutil.unpack_archive(zipfile_path, "/content/unzips", 'zip')
        else:
            return "Arşivden çıkartılacak zip dosyası bulunamadı."
    for root, dirs, files in os.walk('/content/unzips'):
        for file in files:
            if "G_" in file:
                MODELEPOCH = file.split("G_")[1].split(".")[0]
        if MODELEPOCH == '':
            MODELEPOCH = '404'
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".npy") or file.endswith(".index"):
                subprocess.run(["mkdir", "-p", f"/content/RVCCAB/logs/{model}"])
                subprocess.run(["mv", file_path, f"/content/RVCCAB/logs/{model}/"])
            elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                subprocess.run(["mv", file_path, f"/content/RVCCAB/weights/{model}.pth"])
    shutil.rmtree("/content/zips")
    shutil.rmtree("/content/unzips")
    return "Başarıyla tamamlandı."

def download_from_pc(model):
    file_path = model.name
    modelinismi = file_base_name(model.name)

    zip_dirs = ["/content/zips", "/content/unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs("/content/zips", exist_ok=True)
    os.makedirs("/content/unzips", exist_ok=True)
    zipfile_name = model.name
    zipfile_path = '/content/zips/' + zipfile_name
    MODELEPOCH = ''
    shutil.move(file_path, '/content/zips')

    for filename in os.listdir("/content/zips"):
        if filename.endswith(".zip"):
            zipfile_path = os.path.join("/content/zips/", filename)
            with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
                zip_ref.extractall("/content/unzips")
        else:
            return "Arşivden çıkartılacak zip dosyası bulunamadı."
    
    for root, dirs, files in os.walk('/content/unzips'):
        for file in files:
            if "G_" in file:
                MODELEPOCH = file.split("G_")[1].split(".")[0]
            if MODELEPOCH == '':
                MODELEPOCH = '404'
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".npy") or file.endswith(".index"):
                os.makedirs(f"/content/RVCCAB/logs/{modelinismi}", exist_ok=True)
                shutil.move(file_path, f"/content/RVCCAB/logs/{modelinismi}/")
            elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                shutil.move(file_path, f"/content/RVCCAB/weights/{modelinismi}.pth")
    shutil.rmtree("/content/zips")
    shutil.rmtree("/content/unzips")
    return "Başarıyla tamamlandı."



def generate_random_string(length=6):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def list_files_in_current_directory():
    current_directory = os.getcwd()
    files_and_folders = os.listdir(current_directory)
    for item in files_and_folders:
        print(item)


def save_to_wav(dropbox):
    file_path=dropbox.name
    shutil.move(file_path,'/content/RVCCAB/audios')
    return os.path.basename(file_path)


def datasetcreate(file, auto_delete_original_acapella=True, save_to_drive=False):
    file_path = file.name
    shutil.move(file_path, '/content/EasyDataset')
    dataset_name = generate_random_string() 
    os.chdir('/content/EasyDataset')
    for filename in os.listdir():
        if filename.endswith(".wav"):
            sound = AudioSegment.from_wav(filename)
            sound = sound.set_channels(1)
            new_filename = filename
            sound.export('mono_' + new_filename, format="wav")
            os.remove(filename)

    clip_length = 10
    for filename in os.listdir():
        if not filename.endswith('.wav'):
            continue

        wav_file = wave.open(filename, 'rb')
        sample_rate = wav_file.getframerate()
        clip_frames = clip_length * sample_rate
        for i in range(int(wav_file.getnframes() / clip_frames) + 1):
            clip_name = f"{filename.split('.')[0]}_{i+1}.wav"
            clip_path = 'split_' + clip_name
            clip_start = i * clip_frames
            clip_end = min((i + 1) * clip_frames, wav_file.getnframes())

            with wave.open(clip_path, 'wb') as clip_file:
                clip_file.setparams(wav_file.getparams())
                clip_file.writeframes(wav_file.readframes(clip_end - clip_start))

        wav_file.close()
        os.remove(filename)

    os.makedirs(f'/content/dataset/{dataset_name}', exist_ok=True)
    for everything in os.listdir('.'):
        shutil.move(everything, f'/content/dataset/{dataset_name}')

    os.chdir("/content/RVCCAB")

    if auto_delete_original_acapella:
        shutil.rmtree('/content/EasyDataset')
        os.makedirs('/content/EasyDataset', exist_ok=True)
    return(f"Dataset buraya kaydedildi: /content/dataset/{dataset_name} (Lütfen bu yolu kopyalayın sesi eğitirken lazım olacak.)")

def save_models_to_drive():
    logs_dir = './logs/'
    weights_dir = './weights/'
    output_dir = '/content/drive/MyDrive/Finished/'
    finalsavetemp_dir = './finalsavetemp/'

    os.makedirs(output_dir, exist_ok=True)

    pth_files = [file for file in os.listdir(weights_dir) if file.endswith('.pth')]

    skipped_files = set()

    for pth_file in pth_files:
        match = re.search(r'(.*)_s\d+.pth$', pth_file)
        if match:
            base_name = match.group(1)
            if base_name not in skipped_files:
                print(f'Kaydetme geçildi: {base_name}.')
                skipped_files.add(base_name)
            continue

        print(f'Dosya işleniyor: {pth_file}')
        folder = os.path.splitext(pth_file)[0]

        os.makedirs(finalsavetemp_dir, exist_ok=True)
        shutil.copy2(os.path.join(weights_dir, pth_file), finalsavetemp_dir)

        if os.path.isdir(os.path.join(logs_dir, folder)):
            index_files = [
                file for file in os.listdir(os.path.join(logs_dir, folder))
                if file.startswith('added') and file.endswith('.index')
            ]

            if index_files:
                latest_index_file = max(
                    index_files,
                    key=lambda x: os.path.getmtime(os.path.join(logs_dir, folder, x))
                )
                shutil.copy2(
                    os.path.join(logs_dir, folder, latest_index_file),
                    finalsavetemp_dir
                )
                print(f'.index dosyası eşleşti: {latest_index_file}')
            else:
                print(".index dosyası bulunamadı. Lütfen GUI üzerinden Index Dosyasını oluştur butonunu kullanarak index dosyası oluşturun.")
        else:
            print(".index dosyası bulunamadı. Lütfen GUI üzerinden Index Dosyasını oluştur butonunu kullanarak index dosyası oluşturun.")

        # Zip dosyası oluştur ve içine dosyaları yaz
        with zipfile.ZipFile(os.path.join(output_dir, f'{folder}.zip'), 'w') as zipf:
            for root, dirs, files in os.walk(finalsavetemp_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), finalsavetemp_dir))

        print(f'Arşiv dosyası oluşturuldu: {folder}.')

        # 'finalsavetemp' klasörünü sil
        shutil.rmtree(finalsavetemp_dir)

    return 'Yedekleme işlemi bitti. Tamamlanan dosyalar Google Driveda Finished klasörü içerisine kaydedildi. /content/drive/MyDrive/Finished.'

with gr.Blocks(theme=gr.themes.Base(),css="footer {visibility: hidden}", title="RVC.CAB RVC WEB UI",favicon="") as app: 
    gr.HTML("<h1> RVC.CAB Detaylı RVC Arayüzüne Hoş Geldiniz </h1>")
    gr.Markdown(
        value="Bu arayüz <a href='https://github.com/Mangio621/Mangio-RVC-Fork' target='_blank'>Mangio RVC</a> tarafından yapılmış olup <a href='https://rvc.cab' target='_blank'>RVC.CAB</a> tarafından Türkçeleştirilmiş ve sadeleştirilmiştir."
        
    )

    with gr.Tabs():
        
        with gr.TabItem("Ses dönüştürme"):
            # Inference Preset Row
            # with gr.Row():
            #     mangio_preset = gr.Dropdown(label="Inference Preset", choices=sorted(get_presets()))
            #     mangio_preset_name_save = gr.Textbox(
            #         label="Your preset name"
            #     )
            #     mangio_preset_save_btn = gr.Button('Save Preset', variant="primary")

            # Other RVC stuff
            with gr.Row():
                
                #sid0 = gr.Dropdown(label=i18n("推理音色"), choices=sorted(names), value=check_for_name())
                sid0 = gr.Dropdown(label="Modeli seç", choices=sorted(names), value='')
                file_index1 = gr.Textbox(
                    label="Index dosya yolu",
                    value="",
                    interactive=True,
                    visible=False
                )
                            
                file_index2 = gr.Dropdown(
                    label="Index dosyasını seçin",
                    choices=get_indexes(),
                    value=get_index(),
                    interactive=True,
                    allow_custom_value=True
                )
                #input_audio_path2

                
                refresh_button = gr.Button("Model ve ses dosyalarını yenile", variant="primary")

                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("Konuşmacı kimliğini seçin"),
                    value=0,
                    visible=False,
                    interactive=True,
                )

            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        input_audio0 = gr.Textbox(
                            label="Ses dosyası yolu",
                            value=os.path.abspath(os.getcwd()).replace('\\', '/') + "/audios/" + "audio.wav",
                            visible=False
                        )

                        vocalaudiofile = gr.File(
                            file_types=[".mp3", ".wav", ".m4a"],
                            label="Vokal dosyasını yükle (Adında Türkçe karakter ve boşluk olmasın)",
                            show_label=True
                        )
                        vocalaudiofileoutput = gr.Textbox(
                            label="Log kayıtları",
                            interactive=False
                        )
                        vocalaudiofile.upload(save_to_wav, inputs=[vocalaudiofile], outputs=[vocalaudiofileoutput])

                        input_audio1 = gr.Dropdown(
                            label="Ses dosyası seç",
                            choices=sorted(audio_paths),
                            value=get_audios(),
                            interactive=True,
                        )

                    with gr.Column():
                        input_audio1.change(fn=choveraudio,inputs=[],outputs=[input_audio0])
                        f0method0 = gr.Radio(
                            label="Pitch algoritmasını seç",
                            choices=["pm", "harvest", "dio", "crepe", "crepe-tiny", "mangio-crepe", "mangio-crepe-tiny", "rmvpe"], # Fork Feature. Add Crepe-Tiny
                            value="pm",
                            interactive=True,
                        )
                        crepe_hop_length = gr.Slider(
                            minimum=1,
                            maximum=512,
                            step=1,
                            label="Crepe Hop Uzunluğu",
                            value=120,
                            interactive=True,
                            visible=False,
                        )
                        f0method0.change(fn=whethercrepeornah, inputs=[f0method0], outputs=[crepe_hop_length])
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                            value=3,
                            step=1,
                            interactive=True,
                            visible=False
                        )   
                        vc_transform0 = gr.Number(
                            label="Transpoze (pitch) değeri", value=0
                        )
                        but0 = gr.Button("Dönüştür", variant="primary")

                        #sid0.select(fn=match_index, inputs=sid0, outputs=file_index2)
                        
                        

                          
                        refresh_button.click(
                            fn=change_choices, inputs=[], outputs=[sid0, file_index2, input_audio1]
                            )
                        # file_big_npy1 = gr.Textbox(
                        #     label=i18n("特征文件路径"),
                        #     value="E:\\codes\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=0.75,
                            interactive=True,
                            visible=False
                        )
                    with gr.Row():
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                            visible=False
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                            value=0.25,
                            interactive=True,
                            visible=False
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                            visible=False
                        )
                        formanting = gr.Checkbox(
                            value=False,
                            label="[EXPERIMENTAL, WAV ONLY] Formant shift inference audio",
                            info="Used for male to female and vice-versa conversions",
                            interactive=True,
                            visible=False,
                        )
                        
                        formant_preset = gr.Dropdown(
                            value='',
                            choices=get_fshift_presets(),
                            label="browse presets for formanting",
                            visible=False,
                        )
                        formant_refresh_button = gr.Button(value='\U0001f504', visible=False,variant='primary')
                        #formant_refresh_button = ToolButton( elem_id='1')
                        #create_refresh_button(formant_preset, lambda: {"choices": formant_preset}, "refresh_list_shiftpresets")
                        
                        qfrency = gr.Slider(
                                value=Quefrency,
                                label="Quefrency for formant shifting",
                                minimum=-16.0,
                                maximum=16.0,
                                step=0.1,
                                visible=False,
                                interactive=True,
                            )
                        tmbre = gr.Slider(
                            value=Timbre,
                            label="Timbre for formant shifting",
                            minimum=-16.0,
                            maximum=16.0,
                            step=0.1,
                            visible=False,
                            interactive=True,
                        )
                        
                        formant_preset.change(fn=preset_apply, inputs=[formant_preset, qfrency, tmbre], outputs=[qfrency, tmbre])
                        frmntbut = gr.Button("Apply", variant="primary", visible=False)
                        frmntbut.click(fn=formant_apply,inputs=[qfrency, tmbre], outputs=[qfrency, tmbre])
                        formant_refresh_button.click(fn=update_fshift_presets,inputs=[formant_preset, qfrency, tmbre],outputs=[formant_preset, qfrency, tmbre])
                        ##formant_refresh_button.click(fn=preset_apply, inputs=[formant_preset, qfrency, tmbre], outputs=[formant_preset, qfrency, tmbre])
                        ##formant_refresh_button.click(fn=update_fshift_presets, inputs=[formant_preset, qfrency, tmbre], outputs=[formant_preset, qfrency, tmbre])
                    f0_file = gr.File(label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"), visible=False)
                    with gr.Group():
                        vc_output2 = gr.Audio(label=i18n("Sonucu indir"), interactive=True)
                        vc_output1 = gr.Textbox(label="Log Kayıtları")
                    but0.click(
                        vc_single,
                        [
                            spk_item,
                            input_audio0,
                            input_audio1,
                            vc_transform0,
                            f0_file,
                            f0method0,
                            file_index1,
                            file_index2,
                            # file_big_npy1,
                            index_rate1,
                            filter_radius0,
                            resample_sr0,
                            rms_mix_rate0,
                            protect0,
                            crepe_hop_length
                        ],
                        [vc_output1, vc_output2],
                    )
            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0, visible=False
                        )
                        opt_input = gr.Textbox(label=i18n("指定输出文件夹"), value="opt", visible=False)
                        f0method1 = gr.Radio(
                            label=i18n(
                                "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU"
                            ),
                            choices=["pm", "harvest", "crepe", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                            visible=False
                        )
                        
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                            value=3,
                            step=1,
                            interactive=True,
                            visible=False
                        )
                    with gr.Column():
                        file_index3 = gr.Textbox(
                            label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                            value="",
                            interactive=True,
                            visible=False
                        )
                        file_index4 = gr.Dropdown( #file index dropdown for batch
                            label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                            choices=get_indexes(),
                            value=get_index(),
                            interactive=True,
                            visible=False
                        )
                        sid0.select(fn=match_index, inputs=[sid0], outputs=[file_index2, file_index4])
                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4
                        )
                        # file_big_npy2 = gr.Textbox(
                        #     label=i18n("特征文件路径"),
                        #     value="E:\\codes\\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=1,
                            interactive=True,
                            visible=False
                        )
                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                            visible=False
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                            value=1,
                            interactive=True,
                            visible=False
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                            visible=False
                        )
                    with gr.Column():
                        dir_input = gr.Textbox(
                            label=i18n("输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"),
                            value=os.path.abspath(os.getcwd()).replace('\\', '/') + "/audios/",
                            visible=False
                        )
                        inputs = gr.File(
                            file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                            visible=False
                        )
                    with gr.Row():
                        format1 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                            visible=False
                        )
                        but1 = gr.Button(i18n("转换"), variant="primary",visible=False)
                        vc_output3 = gr.Textbox(label=i18n("输出信息"),visible=False)

            sid0.change(
                fn=get_vc,
                inputs=[sid0, protect0, protect1],
                outputs=[],
            )
        with gr.TabItem(i18n("Ses modeli eğit")):
            gr.Markdown(
                value=i18n(
                    "Birkaç temel seçeneği dolduralım. "
                )
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(label=i18n("Model ismini girin"), value="rvccabtest")
                sr2 = gr.Radio(
                    label=i18n("Örnekleme oranı"),
                    choices=["40k", "48k"],
                    value="40k",
                    interactive=True,
                )
                if_f0_3 = gr.Checkbox(
                    label="Whether the model has pitch guidance.",
                    value=True,
                    interactive=True,
                    visible=False
                )
                version19 = gr.Radio(
                    label=i18n("RVC Versiyonunu seç"),
                    choices=["v1", "v2"],
                    value="v1",
                    interactive=True,
                    visible=True,
                )
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label=i18n("Verileri işlemek için kullanılacak işlemci çekirdeği sayısı"),
                    value=int(np.ceil(config.n_cpu / 1.5)),
                    interactive=True,
                )
            with gr.Group():  # 暂时单人的, 后面支持最多4人的#数据处理
                with gr.Row():
                    trainset_dir4 = gr.Textbox(
                        label=i18n("Dataset klasörünün yolunu girin"), value="/content/dataset/"
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=i18n("Konuşmacı ID"),
                        value=0,
                        interactive=True,
                        visible=False
                    )
                    but1 = gr.Button(i18n("Ön işlemeyi başlat"), variant="primary")
                    info1 = gr.Textbox(label=i18n("Log kayıtları"), value="")
                    but1.click(
                        preprocess_dataset, [trainset_dir4, exp_dir1, sr2, np7], [info1]
                    )
            with gr.Group():
                step2b = gr.Markdown("Pitch algoritmasını seçin ardından özellikleri çıkartın")
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(
                            label=i18n("Kullanılacak GPU ID'lerini - ile ayırarak girin. (Burası otomatik doldurulur)"),
                            value=gpus,
                            interactive=True,
                            visible=False
                        )
                        gpu_info9 = gr.Textbox(label=i18n("GPU Bilgisi"), value=gpu_info)
                    with gr.Column():
                        f0method8 = gr.Radio(
                            label=i18n(
                                "Pitch algoritmasını seç"
                            ),
                            choices=["pm", "harvest", "dio", "crepe", "mangio-crepe", "rmvpe"], # Fork feature: Crepe on f0 extraction for training.
                            value="pm",
                            interactive=True,
                        )
                        
                        extraction_crepe_hop_length = gr.Slider(
                            minimum=1,
                            maximum=512,
                            step=1,
                            label="Crepe Hop Uzunluğu",
                            value=64,
                            interactive=True,
                            visible=False,
                        )
                        
                        f0method8.change(fn=whethercrepeornah, inputs=[f0method8], outputs=[extraction_crepe_hop_length])
                    but2 = gr.Button(i18n("Özellikleri çıkart"), variant="primary")
                    info2 = gr.Textbox(label=i18n("Log kayıtları"), value="", max_lines=8)
                    but2.click(
                        extract_f0_feature,
                        [gpus6, np7, f0method8, if_f0_3, exp_dir1, version19, extraction_crepe_hop_length],
                        [info2],
                    )
            with gr.Group():
                gr.Markdown(value=i18n("Son adım! Birkaç ayarı yapmamız gerek."))
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        label=i18n("Kaydetme sıklığı (save_every_epoch)"),
                        value=20,
                        interactive=True,
                        visible=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=1,
                        maximum=10000,
                        step=1,
                        label=i18n("Toplam eğitme sayısı (total_epoch)"),
                        value=500,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        label=i18n("GPU için yığın boyutu"),
                        value=default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Checkbox(
                        label="Whether to save only the latest .ckpt file to save hard disk space",
                        value=True,
                        interactive=True,
                        visible=False
                    )
                    if_cache_gpu17 = gr.Checkbox(
                        label="Tüm dataseti GPU önbelleğine alın. Küçük boyutlu datasetlerde önbelleğe almak hızlandırabilir (en az 10dk) ancak dataset büyüdükçe GPU belleği kullanacağı için performans olumsuz etkilenebilir.",
                        value=False,
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Checkbox(
                        label="Her epoch kaydedildiğinde weights klasörüne son modeli kaydedin",
                        value=True,
                        interactive=True,
                    )
                with gr.Row():
                    pretrained_G14 = gr.Textbox(
                        lines=2,
                        label="Önceden eğitilmiş G model yolu",
                        value="pretrained/f0G40k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        lines=2,
                        label="Önceden eğitilmiş D model yolu",
                        value="pretrained/f0D40k.pth",
                        interactive=True,
                    )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15],
                    )
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, sr2],
                    )
                    ### if f0_3 put here
                    if_f0_3.change(
                            fn=change_f0,
                            inputs=[if_f0_3, sr2, version19, step2b, gpus6, gpu_info9, extraction_crepe_hop_length, but2, info2],
                            outputs=[f0method8, pretrained_G14, pretrained_D15, step2b, gpus6, gpu_info9, extraction_crepe_hop_length, but2, info2],
                    )
                    if_f0_3.change(fn=whethercrepeornah, inputs=[f0method8], outputs=[extraction_crepe_hop_length])
                    gpus16 = gr.Textbox(
                        label=i18n("Kullanılacak GPU ID'lerini - ile ayırarak girin. (Burası otomatik doldurulur)"),
                        value=gpus,
                        interactive=True,
                        visible=False
                    )
                    butstop = gr.Button(
                            "Eğitmeyi durdur",
                            variant='primary',
                            visible=False,
                    )
                    but3 = gr.Button("(1.) Eğitmeye başla", variant="primary", visible=True)
                    but3.click(fn=stoptraining, inputs=[gr.Number(value=0, visible=False)], outputs=[but3, butstop])
                    butstop.click(fn=stoptraining, inputs=[gr.Number(value=1, visible=False)], outputs=[butstop, but3])
                    
                    
                    but4 = gr.Button(i18n("(2.) Index dosyasını oluştur"), variant="primary")
                    info3 = gr.Textbox(label=i18n("Log kayıtları"), value="", max_lines=10)
                    
                    if_save_every_weights18.change(fn=stepdisplay, inputs=[if_save_every_weights18], outputs=[save_epoch10])
                    
                    but3.click(
                        click_train,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            spk_id5,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                        ],
                        [info3, butstop, but3],
                    )
                        
                    but4.click(train_index, [exp_dir1, version19], info3)
                    
                
        with gr.TabItem("İçeri aktar"):
            gr.Markdown(
                value=i18n(
                    "Bu bölümde Google Drive, Mega ya da direkt olarak erişilebilir bir bağlantıdan model dosyalarınızı indirebilir veya vokal dosyanızı içeri aktarabilirsiniz."
                )
            )
            with gr.Column():
                link0 = gr.Textbox(
                    label=i18n("Bağlantı adresi"),
                    interactive=True
                )
                modelismi = gr.Textbox(
                    label=i18n("Model adı (listede gözükecek, dosya ile aynı adı girme zorunluluğunuz yok. TÜRKÇE KARAKTER VE BOŞLUK KULLANMAYIN.)"),
                    interactive=True
                )
                
                downloadBtn = gr.Button(i18n("İndir"), variant="primary")
                
                modeldosyapc = gr.File(
                    file_types=[".zip"],
                    label="Model dosyanızı bilgisayarınızdan yükleyin",
                    show_label=True
                )



            with gr.Column():
                downloadoutput = gr.Textbox(
                    label="Log kayıtları",
                    interactive=False
                )
                downloadBtn.click(
                    download_from_url,
                    [
                        link0,
                        modelismi
                    ],
                    [downloadoutput]
                )
                modeldosyapc.upload(download_from_pc, inputs=[modeldosyapc], outputs=[downloadoutput])

        with gr.TabItem("Dışarı aktar"):
            with gr.Column():
                gr.Markdown("Eğitilen modellerin arşive alıp Google Drive'a aktartabilirsiniz.")
                savebtn = gr.Button(
                    "Drive'a Kaydet",
                    variant="primary"
                )
                savebtnoutput = gr.Textbox(
                    label="Log kayıtları",
                    interactive=False
                )
                savebtn.click(save_models_to_drive, inputs=[], outputs=[savebtnoutput])
        with gr.TabItem("Dataset oluştur"):
            with gr.Group():
                with gr.Column():
                    gr.Markdown("Arkaplan seslerinden ayırdığınız vokal dosyanızı ayıklamak için aşağıdan yükleyin. Yükleme tamamlanır tamamlanmaz ayıklama işlemi başlatılır ve dataset oluşturulur. /content/dataset/ klasörüne rastgele isimle kaydedilir.")
                    datasetcreatedfile = gr.File(
                        label="Vokal dosyası",
                        file_types=[".wav"]
                    )
                    datasetcreateoutput = gr.Textbox(
                        label="Log kayıtları",
                        interactive=False
                    )
                    datasetcreatedfile.upload(datasetcreate, inputs=[datasetcreatedfile], outputs=[datasetcreateoutput])
        with gr.TabItem("Sıkça sorulan sorular"):
            try:
                with open("docs/sss.md", "r", encoding="utf8") as f:
                    info = f.read()
                    gr.Markdown(value=info)
            except:
                gr.Markdown(traceback.format_exc())


    #region Mangio Preset Handler Region
    def save_preset(
        preset_name,
        sid0,
        vc_transform,
        input_audio0,
        input_audio1,
        f0method,
        crepe_hop_length,
        filter_radius,
        file_index1,
        file_index2,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
        f0_file
    ):
        data = None
        with open('../inference-presets.json', 'r') as file:
            data = json.load(file)
        preset_json = {
            'name': preset_name,
            'model': sid0,
            'transpose': vc_transform,
            'audio_file': input_audio0,
            'auto_audio_file': input_audio1,
            'f0_method': f0method,
            'crepe_hop_length': crepe_hop_length,
            'median_filtering': filter_radius,
            'feature_path': file_index1,
            'auto_feature_path': file_index2,
            'search_feature_ratio': index_rate,
            'resample': resample_sr,
            'volume_envelope': rms_mix_rate,
            'protect_voiceless': protect,
            'f0_file_path': f0_file
        }
        data['presets'].append(preset_json)
        with open('../inference-presets.json', 'w') as file:
            json.dump(data, file)
            file.flush()
        print("Saved Preset %s into inference-presets.json!" % preset_name)


    def on_preset_changed(preset_name):
        print("Changed Preset to %s!" % preset_name)
        data = None
        with open('../inference-presets.json', 'r') as file:
            data = json.load(file)

        print("Searching for " + preset_name)
        returning_preset = None
        for preset in data['presets']:
            if(preset['name'] == preset_name):
                print("Found a preset")
                returning_preset = preset
        return (

        )

    if config.iscolab or config.paperspace: # Share gradio link for colab and paperspace (FORK FEATURE)
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True
        )

#endregion
