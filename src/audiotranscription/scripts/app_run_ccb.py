import whisperx
import gc
import json
from pathlib import Path
import torch 
import whisper_helper as helper
import os
os.environ["PATH"] = '/results/projects/ari/transcription_tool/AriAudioTranscription/src/prod_L/src2/ffmpeg-7.0.2-amd64-static:' + os.environ["PATH"]
from datetime import date
from datetime import datetime
import time
import copy 
import numpy as np
import wave
import gc
import transformers
print(transformers.__file__)
import copy
import reconstruction_utils 
import worddoc_utils 
import argparse
import  missingsegment_utils as missingsegments_helper
import shutil



gc.collect()
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser(description="App for speech (audio files) to text")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='dir path of audio files')
    
    parser.add_argument('--output_dir', type=str, default="N/A",
                        help='dir path to save results')
    
    parser.add_argument( '--batch', action='store_true', help='Run in batch mode')
    
    parser.add_argument('--project_name', type=str, default='unknown', 
                        help='project ID')
                        
                        
    parser.add_argument('--project_idpath', type=str, default='unknown', 
                        help='project ID')

    return parser.parse_args()


def generate_compose_transcription(model, audio_file_path, device, batch_size):
    results = helper.get_speechDiar(model=model, 
                                audio_file_path=audio_file_path,  
                                device=device,
                                batch_size=batch_size)

    torch.cuda.empty_cache()
    gc.collect()
    results['segments'] = helper.fix_missingSegmentComponents(results['segments'])

    return results

def get_wavfiles(project_dir):
    audio_files = os.listdir(project_dir)
    generated_files=[]
    # new_audio_files = []
    for file_ in audio_files:
        if file_.split('.')[-1] in ['mp4','m4a']:
            input_file = os.path.join(project_dir, file_)
            output_name=  f'''{file_.split('.')[0]}.wav'''
            output_file = os.path.join(project_dir, output_name)
            helper.create_wav(input_file, output_file)
            print (f'The following file was generated: {output_file}')
            generated_files.append(output_file)
            # new_audio_files.append(output_name)
        # else:
        #     if file_.split('.')[-1] == 'wav':
        #          print (f'The following file was found: {output_file}')
        #          output_file = os.path.join(project_dir, file_)
        #          generated_files.append(output_file)
    # return new_audio_files
    return generated_files

def generate_individual_transcription(model, project_dir, device,batch_size):
    
    audio_files = os.listdir(project_dir)
    audio_files = [i for i in audio_files if '.wav' in i ]

    
    ## remove master
    audio_files = [ i for i in audio_files if i.split('.')[0].lower() != 'compose']
    audio_files.sort()
    print (f'The following individual/speaker files will be processed: {audio_files}')
    speaker_stt={}
    for audio_file in audio_files:
            audio_file_path = os.path.join(project_dir, audio_file)
            result = helper.get_speechDiar(model=model, 
                                            audio_file_path=audio_file_path,  
                                            device=device,
                                            batch_size=batch_size)
            
            gc.collect()
            torch.cuda.empty_cache()
            speaker_name = helper.get_speaker_name(audio_file)
            speaker_stt[speaker_name] = result.copy()
    return speaker_stt

def clean_speakers_stt(speaker_stt):
    for speaker in list(speaker_stt.keys()):
        if len(speaker_stt[speaker]['segments']) == 0:
            print (f'{speaker} speaker was deleted. No speech was found')
            del speaker_stt[speaker]
    return speaker_stt

def get_projectID(file_path):
  with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()
  
  return content.strip()

def get_transcripts(output_dir, project_dir, model,device, project_name):
        
        ## Create output dir
        os.makedirs(output_dir, exist_ok=True)

        ## Create temporal dir
        temporal_dir = os.path.join(output_dir, 'temp')
        os.makedirs(temporal_dir, exist_ok=True)

        ## Create missingSegments dir
        missing_dir =os.path.join(output_dir, 'speaker_missing_segments')
        os.makedirs(missing_dir, exist_ok=True) 
        
        
        ## create wav files if they do not exist
        generated_files = get_wavfiles(project_dir)

        ## Get compose file:
        compose_file_path = os.path.join(project_dir, 'compose.wav')
        if os.path.exists(compose_file_path) ==False:
            print (f'ERROR. No comose file was found in {project_dir}. Please, provide compose audio file')
            return 
    
        start_time = time.time()
        ## Generate composite transcription
        batch_size= 16
        results = generate_compose_transcription(model=model, 
                                                audio_file_path= compose_file_path, device=device, batch_size= batch_size)
    
        ## save whisper output, temp files
        output_file_name = os.path.join(temporal_dir, f'whisperx_{project_name}.txt')
        result_edited = copy.deepcopy(results)
        result_edited = helper.fix_result_format(result_edited)
        result_edited = helper.fix_diar(result_edited)
        with open( output_file_name, "w", encoding="utf-8-sig") as f:
            helper.get_speaker_aware_transcript(result_edited, f)


        ### process individual files
        speaker_stt  = generate_individual_transcription(model=model, project_dir=project_dir, device=device, batch_size=batch_size)
        if len(speaker_stt)==0:
            print (f'ERROR. No speaker files were found in {project_dir}')
            return 
        ## clean empty speakers
        speaker_stt = clean_speakers_stt(speaker_stt)
        end_time = time.time()
        total_time1 = end_time - start_time
        print(f'Transcription execution time: {end_time - start_time:.4f} seconds')

        ## change speaker names of segments
        speaker_stt  = helper.change_SegmentSpeakerNames(speaker_stt)
        SPEAKERS = list(speaker_stt.keys())
        print (f'The following speakers were processed: {SPEAKERS}')

        print ('Working on Reconstruction...')
        results, LOG = reconstruction_utils.reconstruction_v2(master_result = results, speaker_stt= speaker_stt)
        print ('Reconstruction finished!')

        ## save transcript 
        output_fileName = os.path.join(temporal_dir,f'reconstruction-{project_name}.txt')
        with open( output_fileName, "w", encoding="utf-8-sig") as f:
                       reconstruction_utils.get_speaker_aware_transcript(results, f)
        docx_path =  os.path.join(output_dir,f'reconstruction-{project_name}.docx')
        worddoc_utils.create_word_document_transcriptColored(docx_path,output_fileName)
        print (f'The final transcript was saved at {docx_path}')

        ## save transcript without time stamp
        output_fileName = os.path.join(temporal_dir,f'reconstruction-{project_name}-NTS.txt')
        with open( output_fileName, "w", encoding="utf-8-sig") as f:
                       reconstruction_utils.get_speaker_aware_transcript(results, f, timestamp=False)

        docx_path =  os.path.join(output_dir,f'reconstruction-{project_name}-NTS.docx')
        worddoc_utils.create_word_document_transcriptColored_noTimeStamp(docx_path,output_fileName)
        print (f'The final transcript was saved at {docx_path}')

        end_time = time.time()
        total_time2 = end_time - start_time
        print(f'Total execution time: {end_time - start_time:.4f} seconds')

        LOG['transcript_path'] = docx_path
        LOG['transcription_executionTime'] = f'{round(total_time1/60,2)} mins'    
        LOG['total_executionTime'] = f'{round(total_time2/60,2)} mins'

    
        ## report missing segments (speaker segments that were not assigned)
        output_fileName_withMissing = os.path.join(temporal_dir,f'reconstruction-wMissing-{project_name}.txt')
        output_fileName_withMissingNTS = os.path.join(temporal_dir,f'reconstruction-{project_name}-wMissing-NTS.txt')

        df_m = missingsegments_helper.get_missing_segments(pred_result = results, speaker_stt = speaker_stt, 
                                                           output_fileName_withMissing= output_fileName_withMissing ,
                                                           output_fileName_withMissingNTS =  output_fileName_withMissingNTS)

        
        
        ## create a word file for edited transcript with missing segments 
        docx_path =  os.path.join(output_dir,f'reconstruction--wMissing-{project_name}.docx')
        worddoc_utils.create_word_document_transcriptColored(docx_path,output_fileName_withMissing)
        print (f'The transcript with speaker-missing segments was saved at {docx_path}')    


        ## create a word file for edited transcript with missing segments with no time stamps
        docx_path =  os.path.join(output_dir,f'reconstruction--wMissing-{project_name}-NTS.docx')
        worddoc_utils.create_word_document_transcriptColored_noTimeStamp(docx_path,output_fileName_withMissingNTS)
        print (f'The transcript with speaker-missing segments was saved at {docx_path}') 

        LOG['Total_n_missingSegments [from individual files]'] = df_m.shape[0]
        ## create a docx with the table of missing segments
        worddoc_utils.df_to_docx_table(df = df_m, filepath =os.path.join(missing_dir,f'MissingSegmentsTable-{project_name}.docx'))
        for speaker in speaker_stt:
            df = df_m[df_m['speaker']==speaker]
            LOG[f'Total_n_missingSegments_{speaker}'] = df.shape[0]
            worddoc_utils.df_to_docx_table(df = df, filepath =os.path.join(missing_dir,f'MissingSegmentsTable-{project_name}-{speaker}.docx'))


        print (f'Log of Events: /n {LOG}')

        ## save LOG
        json_path = os.path.join(output_dir, f'log_{project_name}.json')        
        with open(json_path, 'w', encoding='utf-8') as f:
          json.dump(LOG, f, ensure_ascii=False, indent=4)      
        print (f'The LOG can be found at {json_path}')

        ## delete wav-generated files
        for file_path in generated_files:
            os.remove(file_path)
            print (f'''The following file was eliminated: {file_path}''')
            
        ## delete all files from input directory
        for file_ in os.listdir(project_dir):
            file_path =os.path.join(project_dir,file_)
            if Path(file_path).is_file():
              os.remove(file_path)
              print (f'''The following file was eliminated: {file_path}''')      
        
def main():
        compute_type = "float16"
        batch_size = 16 # reduce if low on GPU mem
        device = "cuda"
        model = whisperx.load_model("large-v2", device, compute_type=compute_type, language = 'en')
        
        args = parse_args()
        if args.batch:
            print ('Batch processing was found...')
            objs = os.listdir(args.input_dir)
            for obj in objs:
                project_dir = os.path.join(args.input_dir, obj )
                if Path(project_dir).is_dir():
                    output_dir = args.output_dir
                    project_name = project_dir.split('/')[-1]
                    print (f'processing {project_name}')
                    if output_dir =="N/A":
                        output_dir = os.path.join(project_dir, 'output_files')
                    else:
                        output_dir = os.path.join(output_dir, project_name, 'output_files')
                        os.makedirs(output_dir, exist_ok=True)                        
                
                    get_transcripts(output_dir, project_dir, model,device, project_name)
                    shutil.rmtree(project_dir)
        else:
            project_dir = args.input_dir
            project_name =  args.project_name
            if project_name == 'unknown':
                project_id_path = args.project_idpath
                if os.path.exists(project_id_path):
                    project_name = get_projectID(project_id_path)
                else:         
                    project_name = date.today().strftime("%Y-%m-%d")+'_' + datetime.now().time().strftime("%H:%M:%S") #date.today().strftime("%Y-%m-%d")
                    project_name = project_name.replace(':', '-')
            output_dir = args.output_dir
            if output_dir =="N/A":
                output_dir = os.path.join(project_dir, 'output_files')

            get_transcripts(output_dir=output_dir, 
                            project_dir= project_dir, 
                            model = model,
                            device = device, 
                            project_name = project_name)

if __name__ == "__main__":
    main()