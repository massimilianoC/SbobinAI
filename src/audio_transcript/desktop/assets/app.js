'use strict';
const $ = id => document.getElementById(id);
let step = 1, selected = null, queueData = null, peaks = null, busy = false, running = false, timer = null;
let startTime = 0, endTime = 0;
let fullRetention = 'keep-all', defaultRetention = 'keep-all', waveToken = 0;
let pb = null, candidate = -1, cueIndex = -1, mp4Reason = '';
const formatLabels = {'mp4':'MP4 720p','txt':'TXT','md':'Markdown','srt':'SRT','vtt':'VTT','json':'Transcript JSON','mp3':'MP3','flac':'FLAC','report-md':'Report MD','report-json':'Report JSON'};
const outputFormats = () => [...document.querySelectorAll('[data-output]:checked')].map(input=>input.dataset.output);
const formatTime = seconds => {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 3600).toString().padStart(2,'0')}:${Math.floor(s / 60 % 60).toString().padStart(2,'0')}:${(s % 60).toString().padStart(2,'0')}`;
};
function error(message) { $('alert').textContent = message; $('alert').hidden = !message; }
function controls() {
  $('next').disabled = busy || running || (step === 1 && !selected) || (step === 3 && !outputFormats().length);
  $('back').disabled = busy || running;
  for (const id of ['import','refresh','whole','start']) $(id).disabled = busy || running;
  document.querySelectorAll('input[name=file]').forEach(input => input.disabled = busy || running);
  document.querySelectorAll('[data-output]').forEach(input => input.disabled = busy || running);
  $('mp4').disabled = busy || running || !!mp4Reason;
  for (const id of ['language','context','start-range','end-range','start-time','end-time']) $(id).disabled = busy || running;
  $('retention').disabled = busy || running || !selected || startTime !== 0 || Math.abs(endTime-selected.duration) >= .001;
  $('back').hidden = step === 1 || step >= 5 || running;
  $('next').hidden = step >= 4;
}
async function call(name, ...args) {
  const result = await window.pywebview.api[name](...args);
  return result;
}
async function task(action) {
  if (busy || running) return;
  busy = true; controls(); error('');
  try { return await action(); }
  catch (e) { error(e.message || String(e)); }
  finally { busy = false; controls(); }
}
function show(n) {
  step = n;
  document.querySelectorAll('[data-step]').forEach(el => el.hidden = Number(el.dataset.step) !== n);
  [...$('steps').children].forEach((el, i) => {
    el.classList.toggle('active', i + 1 === n); el.classList.toggle('done', i + 1 < n);
    if (i + 1 === n) el.setAttribute('aria-current','step'); else el.removeAttribute('aria-current');
  });
  document.body.dataset.view = n;
  controls(); if (n === 3) requestAnimationFrame(draw); if (n === 6) requestAnimationFrame(drawReview);
}
function rows(target, values) {
  target.replaceChildren();
  for (const [name,value] of values) {
    const row = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = name; dd.textContent = value; row.append(dt,dd); target.append(row);
  }
}
async function choose(id) {
  selected = null; peaks = null; waveToken++; controls();
  selected = await call('select', id); startTime = 0; endTime = selected.duration;
  defaultRetention = selected.default_after_success; fullRetention = defaultRetention;
  mp4Reason = !selected.has_video ? 'Audio-only recording: there is no video to render.' : !selected.mp4_encoder ? 'This FFmpeg build cannot render MP4 (libx264 is missing).' : '';
  if (mp4Reason) $('mp4').checked = false;
  mp4Note(); controls();
  $('selection-info').textContent = `${selected.name} · ${formatTime(selected.duration)}`;
  for (const id of ['start-range','end-range','start-time','end-time']) $(id).max = selected.duration;
  updateInterval();
  loadWaveform();  // background: never awaited, never sets busy
}
async function renderQueue(data, preferred) {
  queueData = data; selected = null; peaks = null; waveToken++; $('files').replaceChildren();
  $('file-count').textContent = `${data.files.length} recording${data.files.length === 1 ? '' : 's'}`;
  $('selection-info').textContent = 'Choose a recording to begin';
  for (const file of data.files) {
    const label = document.createElement('label'); label.className = 'file-card'; label.style.margin = '0';
    const radio = document.createElement('input'); radio.type = 'radio'; radio.name = 'file'; radio.value = file.id;
    const content = document.createElement('div'), name = document.createElement('div'), size = document.createElement('div');
    name.className='file-name'; name.textContent=file.name; size.className='file-size'; size.textContent=`${(file.bytes / 1024 / 1024).toFixed(1)} MB · ${file.name.split('.').pop().toUpperCase()}`;
    content.append(name,size); label.append(radio,content); $('files').append(label);
    radio.addEventListener('change', () => task(async () => { await choose(file.id); }));
  }
  if (!data.files.length) {
    const empty = document.createElement('p'); empty.className='hint'; empty.textContent='Your input queue is empty. Import a recording to get started.'; $('files').append(empty);
  }
  const id = preferred ?? (data.files.length === 1 ? data.files[0].id : null);
  if (id !== null) {
    [...$('files').querySelectorAll('input')].find(input => input.value === id).checked = true;
    await choose(id);
  }
  controls();
}
function updateInterval() {
  if (!selected) return;
  $('start-range').value = startTime; $('end-range').value = endTime;
  $('start-time').value = Number(startTime.toFixed(3)); $('end-time').value = Number(endTime.toFixed(3));
  const full = startTime === 0 && Math.abs(endTime - selected.duration) < .001;
  $('selected-length').textContent = `${full ? 'Full recording' : 'Selected interval'} · ${formatTime(endTime - startTime)}`;
  $('retention').disabled = !full || busy || running;
  $('retention').value=full?fullRetention:'keep-all';
  $('retention-note').textContent = full ? retentionNote() : 'This is a partial transcription. The queued recording will stay in input and no source will be deleted.';
  draw();
}
function mp4Note() {
  $('mp4-note').hidden = !(mp4Reason || $('mp4').checked);
  $('mp4-note').textContent = mp4Reason || 'MP4 is rendered after transcription, on the GPU when available, otherwise on the CPU; long videos take a while.';
}
function retentionNote() {
  const imported = defaultRetention === 'delete-all', del = $('retention').value === 'delete-all';
  if (imported) return del ? 'The imported copy in input is deleted only after a complete transcription and verified outputs. The original you chose in the picker stays untouched.' : 'The imported copy is archived in processed. The original you chose in the picker stays untouched.';
  return del ? 'Warning: this file was already in input, so the app cannot know whether another copy exists. Deleting it after success may remove the only copy.' : 'This file was already in input. After a complete transcription it is archived in processed, not deleted.';
}
function setWave(state) { $('wave-panel').dataset.state = state; $('wave-failed').hidden = state !== 'failed'; }
async function loadWaveform() {
  const token = ++waveToken; peaks = null; setWave('loading'); draw();
  try {
    const data = await call('preview');
    if (token !== waveToken) return;  // a newer selection owns the display
    if (data.cancelled) setWave('failed'); else { peaks = data.peaks; setWave('ready'); }
  } catch (e) { if (token !== waveToken) return; setWave('failed'); }
  draw();
}
function draw() {
  const canvas = $('waveform'), width = canvas.clientWidth, height = canvas.clientHeight;
  if (!width || !selected) return;
  const scale = window.devicePixelRatio || 1; canvas.width=width*scale; canvas.height=height*scale;
  const ctx=canvas.getContext('2d'); ctx.scale(scale,scale);
  const a=startTime/selected.duration*width, b=endTime/selected.duration*width;
  ctx.fillStyle='#eef0ea'; ctx.fillRect(0,0,width,height-20);
  ctx.fillStyle='#e0eddc'; ctx.fillRect(a,0,b-a,height-20);
  if (peaks) {
    const max=Math.max(...peaks,.01);
    peaks.forEach((value,i) => {
      const x=i/peaks.length*width, h=Math.max(1,value/max*(height-30));
      ctx.fillStyle=x>=a && x<=b ? '#397859' : '#b9c1b8';
      ctx.fillRect(x,(height-20-h)/2,Math.max(.7,width/peaks.length*.75),h);
    });
  } else {
    ctx.strokeStyle='#c3cec0'; ctx.beginPath(); ctx.moveTo(0,(height-20)/2); ctx.lineTo(width,(height-20)/2); ctx.stroke();
  }
  ctx.strokeStyle='#397859'; [a,b].forEach(x=> {ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,height-20);ctx.stroke();});
  ctx.font='9px Segoe UI, sans-serif';ctx.fillStyle='#72816f';
  for(let i=0;i<=4;i++){ctx.textAlign=i===0?'left':i===4?'right':'center';ctx.fillText(formatTime(selected.duration*i/4),width*i/4,height-3);}
}
function payload() { return {language:$('language').value,context:$('context').value,start:startTime,end:endTime,after_success:$('retention').value,output_formats:outputFormats()}; }
async function review() {
  const recap=await call('review',payload());
  rows($('recap'),[['Recording',recap.file],['Language',$('language').selectedOptions[0].textContent],['Context',$('context').value || 'None'],['Interval',`${formatTime(startTime)} → ${formatTime(endTime)} (${formatTime(endTime-startTime)})`],['Queued file',$('retention').selectedOptions[0].textContent],['Outputs',outputFormats().map(key=>formatLabels[key]).join(' · ')],['Model',recap.model]]);
  $('recap-panel').open=true;
  $('recap-brief').textContent=`${formatTime(endTime-startTime)} · ${outputFormats().length} formats`;
  $('start').textContent=$('retention').value==='delete-all'?'Transcribe & delete queued file after success →':'Start transcription →';
}
const phaseLabels={starting:'Starting the model',probe:'Inspecting recording',preparation:'Preparing audio',inference:'Transcribing',export:'Writing transcripts',audio_export:'Preparing selected audio',video_export:'Rendering preview video',archive:'Managing the queued file',packaging:'Packaging intermediate exports',catalog:'Updating catalog',reuse:'Reusing the existing transcription: no model work needed',finished:'Finished'};
const stageIndexes={starting:0,probe:1,preparation:1,inference:2,export:3,audio_export:3,video_export:3,reuse:3,archive:4,packaging:4,catalog:4,finished:5};
const encoderLabel=r=>r.encoder==='h264_nvenc'?(r.gpu_decode?'NVENC · GPU':'NVENC'):'CPU';
const afterInference=['export','audio_export','archive','packaging','catalog'];
const stageLabels=[['probe','Inspect'],['audio_extraction','Audio extraction'],['speech_detection','Speech detection'],['chunk_slicing','Chunking'],['inference','Transcription'],['export','Exports'],['audio_export','Audio'],['video_export','Preview video'],['archive','File management'],['packaging','Packaging'],['catalog','Catalog']];
const duration=s=>{s=Number(s);return s<10?`${s.toFixed(1)} s`:s<60?`${Math.round(s)} s`:formatTime(s);};
const sizeText=b=>b>=1048576?`${(b/1048576).toFixed(1)} MB`:`${Math.max(1,Math.round(b/1024))} KB`;
const gbText=mb=>`${(mb/1024).toFixed(1)}`;
function card(id, percent, number, sub) {
  const el=document.querySelector(`[data-res=${id}]`);
  el.classList.toggle('none', number==='—'); el.classList.toggle('hot', percent!=null && percent>90);
  el.querySelector('.res-num').textContent=number; el.querySelector('.res-sub').textContent=sub||'';
  el.querySelector('.res-bar i').style.width=`${Math.max(0,Math.min(100,percent||0))}%`;
}
function updateCards(r, phase) {
  const pct=v=>v!=null?`${Math.round(v)}%`:'—', ratio=(u,t)=>u!=null&&t?u/t*100:null;
  card('cpu', r.cpu_percent, pct(r.cpu_percent), '');
  const sub=[r.gpu_temp_c!=null?`${Math.round(r.gpu_temp_c)} °C`:'']; if(phase==='video_export' && r.gpu_encoder_percent!=null) sub.unshift(`Encoder ${Math.round(r.gpu_encoder_percent)} %`);
  card('gpu', r.gpu_util_percent, pct(r.gpu_util_percent), sub.filter(Boolean).join(' · '));
  card('vram', ratio(r.gpu_mem_used_mb,r.gpu_mem_total_mb), r.gpu_mem_used_mb!=null?`${gbText(r.gpu_mem_used_mb)} GB`:'—', r.gpu_mem_total_mb!=null?`of ${gbText(r.gpu_mem_total_mb)} GB`:'');
  card('ram', ratio(r.ram_used_mb,r.ram_total_mb), r.ram_used_mb!=null?`${gbText(r.ram_used_mb)} GB`:'—', r.ram_total_mb!=null?`of ${gbText(r.ram_total_mb)} GB`:'');
}
function bar(percent, detail) {
  if(percent!=null) $('progress').value=percent; else $('progress').removeAttribute('value');
  $('percent').textContent=percent!=null?`${percent.toFixed(0)}%`:'…'; $('progress-detail').textContent=detail;
}
async function poll() {
  try {
    const status=await call('status'), progress=status.progress || {}, step_progress=status.stage_progress, phase=status.phase, reused=!!status.reused;
    $('phase').textContent=phaseLabels[phase] || phase; $('reuse-note').hidden=!reused;
    const chunkPercent=Math.min(100,Math.max(0,Number(progress.percent || 0)));
    if(phase==='video_export' && step_progress) {
      const p=step_progress.percent, eta=step_progress.eta_s, speed=step_progress.speed_x;
      bar(p!=null?p:null,`Rendering preview video${p!=null?` · ${Math.round(p)} %`:''}${eta!=null?` · about ${formatTime(eta)} remaining`:''}${speed!=null?` · ${Number(speed).toFixed(1)}× realtime`:''}${step_progress.encoder?` · ${encoderLabel(step_progress)}`:''}`);
    } else if(phase==='video_export') bar(null,'Rendering preview video · starting…');
    else if(phase==='inference' && progress.total_chunks != null) bar(chunkPercent,`${progress.done_chunks || 0} / ${progress.total_chunks} chunks · ${progress.eta_s != null ? 'About '+formatTime(progress.eta_s)+' remaining' : 'Estimating remaining time…'}`);
    else if(afterInference.includes(phase) && !reused && progress.total_chunks != null) bar(100,'Transcription finished · writing the selected outputs…');
    else bar(null,reused?'No model work needed · preparing the selected outputs…':'Preparing the recording…');
    const messages=[...(status.messages || [])];
    if(status.runtime) messages.unshift(`Backend: ${status.runtime.backend} · ${status.runtime.device_name || status.runtime.device}${status.runtime.fallback_used?' · FALLBACK IN USE':''}`);
    $('messages').textContent=messages.join('\n');
    $('messages').scrollTop=$('messages').scrollHeight;
    const rendering=phase==='video_export' && step_progress;
    $('metric-chunks').textContent=reused?'reused':progress.total_chunks != null?`${progress.done_chunks || 0} / ${progress.total_chunks}`:'—';
    $('metric-eta').textContent=rendering&&step_progress.eta_s!=null?formatTime(step_progress.eta_s):progress.eta_s != null&&!reused?formatTime(progress.eta_s):reused?'—':'Estimating…';
    $('metric-speed').textContent=rendering&&step_progress.speed_x!=null?`${Number(step_progress.speed_x).toFixed(1)}× realtime`:progress.throughput_x != null&&!reused?`${Number(progress.throughput_x).toFixed(1)}× realtime`:'—';
    $('metric-device').textContent=status.runtime?`${status.runtime.backend.toUpperCase()}${status.runtime.fallback_used?' · fallback':''}`:'Detecting…';
    updateCards(status.resources || {}, phase);
    const stageIndex=stageIndexes[phase] ?? 0;
    [...$('stage-rail').children].forEach((el,index)=>{const skipped=reused&&index<3;el.classList.toggle('active',index===stageIndex);el.classList.toggle('done',index<stageIndex&&!skipped);el.classList.toggle('reused',skipped);});
    if(!status.running && status.result) {
      running=false; clearInterval(timer); timer=null; finish(status); controls();
    }
  } catch(e) { error(`Could not read progress: ${e.message || e}. The worker may still be processing; keep this window open.`); }
}
function finish(status) {
  const result=status.result, summary=status.summary || {}, ok=result.ok, reused=!!status.reused;
  document.body.classList.remove('processing');
  $('result-mark').textContent=ok?'✓':'!';
  $('result-title').textContent=ok?'Your outputs are ready.':'This recording needs attention.';
  $('result-description').textContent=ok?(reused?'An earlier transcription with these exact settings was reused, so the model did no work this time.':'Open the result folder, or review the recording with its subtitles.'):result.jobs?.[0]?.error || summary.error || result.error?.message || 'The recording remains queued. Review the local report and re-run to retry missing chunks.';
  const statusText=(summary.status==='skipped'?'Completed (reused)':summary.status || (ok?'Completed':'Failed')).replace(/^./,c=>c.toUpperCase());
  const counts=summary.counts || {}, facts=[['Recording',selected.name],['Status',statusText]];
  if(status.runtime && !reused) facts.push(['Backend',`${status.runtime.backend.toUpperCase()} · ${status.runtime.device_name || status.runtime.device}${status.runtime.fallback_used?' · fallback':''}`]);
  else if(reused) facts.push(['Transcription',status.reused.completed_at?`Reused from ${status.reused.completed_at.slice(0,10)}`:'Reused from an earlier run']);
  if(status.progress?.throughput_x != null && !reused) facts.push(['Speed',`${Number(status.progress.throughput_x).toFixed(1)}× realtime`]);
  if(counts.total_chunks != null) facts.push(['Chunks',`${counts.ok || 0} transcribed · ${counts.no_speech || 0} silent · ${counts.failed || 0} failed`]);
  const analysed=status.prepared?.analysed_seconds;
  facts.push(['Duration',analysed!=null?`${formatTime(analysed)}${status.prepared.speech_seconds!=null?` · speech ${formatTime(status.prepared.speech_seconds)}`:''}`:formatTime(endTime-startTime)]);
  facts.push(['Queued file','Checking…']);
  rows($('result-summary'),facts);
  $('outcome-brief').textContent=`${statusText} · checking the queued file…`; $('outcome-brief').dataset.status=statusText;
  $('outcome-panel').open=!ok; $('times-panel').open=false;
  const times=[], stages=status.stages || {};
  for(const [key,label] of stageLabels) if(stages[key] != null) times.push([label,duration(stages[key])]);
  const wall=summary.timings?.wall_seconds;
  if(wall != null) times.push(['Total',duration(wall)]);
  rows($('result-times'),times.length?times:[['Timings','Not available']]);
  const last=$('result-times').lastElementChild; if(wall != null && last) last.className='total';
  $('times-brief').textContent=wall!=null?`Total ${duration(wall)}${reused?' · transcription reused':''}`:'Not available';
  const original=status.reused, note=$('result-original'); note.hidden=!original;
  if(original) note.textContent=`Original transcription${original.completed_at?` (completed ${original.completed_at.slice(0,10)})`:''}: ${stageLabels.filter(([k])=>original.stages?.[k]!=null).map(([k,l])=>`${l} ${duration(original.stages[k])}`).join(' · ')}${original.wall_seconds!=null?` · total ${duration(original.wall_seconds)}`:''}`;
  $('open-result').disabled=!status.can_open; $('review-open').hidden=!status.can_review;
  $('files-summary').textContent='Checking files…'; $('files-summary').className=''; $('files-table').tBodies[0].replaceChildren();
  if(status.can_open) call('result_details').then(showFiles).catch(e=>{$('files-summary').textContent=`Could not check the files: ${e.message || e}`;});
  else $('files-summary').textContent='No result folder to check.';
  if(result.jobs?.[0]?.delivery_warning) $('result-description').textContent+=' '+result.jobs[0].delivery_warning;
  show(5);
}
function showFiles(details) {
  const body=$('files-table').tBodies[0]; body.replaceChildren();
  for(const file of details.files) {
    const row=body.insertRow(), cells=[file.name,file.kind,sizeText(file.size)];
    cells.forEach((text,i)=>{const cell=row.insertCell(); cell.textContent=text; if(i)cell.className=i===1?'kind':'size';});
    const mark=document.createElement('span'); mark.className=`mark ${file.state}`;
    mark.textContent=({verified:'✓ Verified',complete:'✓ Complete',partial:'Partial',problem:'✕ Problem'})[file.state] || file.state;
    row.insertCell().append(mark); const note=row.insertCell(); note.className='note'; note.textContent=file.note;
  }
  const bad=details.problems;
  $('files-summary').textContent=!details.total?'No files found.':bad?`${bad} file${bad===1?'':'s'} need${bad===1?'s':''} attention`:`All ${details.total} files checked`;
  $('files-summary').className=bad?'bad':'';
  if(details.queued) { $('outcome-brief').textContent=`${$('outcome-brief').dataset.status} · ${details.queued.text.replace(/\.$/,'')}`; }
  if(details.queued) { const dd=[...$('result-summary').children].find(r=>r.firstChild.textContent==='Queued file')?.lastChild; if(dd) dd.textContent=details.queued.text; }
}
$('import').onclick=()=>task(async()=>{const result=await call('import_file');if(!result.cancelled) await renderQueue(result,result.imported_id);});
$('refresh').onclick=()=>task(async()=>renderQueue(await call('queue')));
$('wave-retry').onclick=loadWaveform;
$('whole').onclick=()=>{startTime=0;endTime=selected.duration;updateInterval();};
for(const id of ['start-range','start-time']) $(id).addEventListener(id.endsWith('time')?'change':'input',()=>{const v=Number($(id).value);if(Number.isFinite(v)){startTime=Math.max(0,Math.min(v,endTime-.05));updateInterval();}});
for(const id of ['end-range','end-time']) $(id).addEventListener(id.endsWith('time')?'change':'input',()=>{const v=Number($(id).value);if(Number.isFinite(v)){endTime=Math.min(selected.duration,Math.max(v,startTime+.05));updateInterval();}});
$('back').onclick=()=>{error('');show(step-1);};
$('next').onclick=()=>task(async()=>{if(step===3)await review();show(step+1);});
$('start').onclick=()=>task(async()=>{await call('start',payload());running=true;$('resource-cards').hidden=false;$('res-note').hidden=false;document.body.classList.add('processing');$('recap-panel').open=false;$('start').hidden=true;$('run-progress').hidden=false;$('run-heading').textContent='Making words from sound.';$('run-intro').textContent='Processing locally. Your configuration stays available above.';timer=setInterval(poll,800);await poll();});
$('open-result').onclick=()=>task(async()=>call('open_result'));
$('again').onclick=()=>task(async()=>{error('');await releasePlayback();$('resource-cards').hidden=true;$('res-note').hidden=true;$('start').hidden=false;$('recap-panel').open=true;$('run-progress').hidden=true;$('run-heading').textContent='Everything ready?';$('run-intro').textContent='Review your choices, then start the local transcription.';await renderQueue(await call('queue'));show(1);});
$('retention').onchange=()=>{fullRetention=$('retention').value;updateInterval();};
document.querySelectorAll('[data-output]').forEach(input=>input.addEventListener('change',()=>{mp4Note();controls();}));
new ResizeObserver(draw).observe($('waveform'));
// Optional step 6: review playback with live subtitles (GUI-09).
const player=$('player'), currentCandidate=()=>pb?.candidates[candidate];
async function openReview() {
  pb=await call('playback'); cueIndex=-1; $('subtitle').textContent='';
  $('cues').replaceChildren(...pb.cues.map(q=>{
    const li=document.createElement('li'), time=document.createElement('time'), text=document.createElement('span');
    time.textContent=formatTime(q.start); text.textContent=q.text; li.tabIndex=0; li.append(time,text);
    const seek=()=>{const c=currentCandidate(); if(c) player.currentTime=c.offset+q.start;};
    li.onclick=seek; li.onkeydown=e=>{if(e.key==='Enter')seek();}; return li;
  }));
  show(6); await playFrom(0);
}
async function playFrom(index) {
  candidate=index; const c=currentCandidate();
  if(!c) { $('review-source').textContent=index?'None of the available recordings can be played in this window (for example, a video codec the system web view cannot decode). The cues are listed below; open the result folder to play the file in another application.':'The original and the exported audio are no longer available, so there is nothing to play. The cues are listed below.'; return; }
  try { await call('playback_select',index); } catch(e) { return playFrom(index+1); }
  $('review-source').textContent=`Playing ${c.label}.`+(index?' The first source could not be decoded here.':'');
  player.src=c.url;
}
async function releasePlayback() {
  pb=null; candidate=-1; player.pause(); player.removeAttribute('src'); player.load(); $('subtitle').textContent='';
  await call('playback_stop');
}
function drawReview() {
  const canvas=$('review-wave'); canvas.hidden=!(pb&&peaks&&selected); if(canvas.hidden) return;
  const width=canvas.clientWidth, height=canvas.clientHeight, scale=window.devicePixelRatio||1, c=currentCandidate(), axis=16, area=height-axis;
  canvas.width=width*scale; canvas.height=height*scale; const ctx=canvas.getContext('2d'); ctx.scale(scale,scale);
  const n=peaks.length, d=selected.duration, from=Math.floor(pb.selection_start/d*n), to=Math.min(n,Math.max(from+1,Math.ceil((pb.selection_start+pb.selection_duration)/d*n)));
  const part=peaks.slice(from,to), max=Math.max(...part,.01), head=c?(player.currentTime-c.offset)/pb.selection_duration*width:-1;
  part.forEach((value,i)=>{const x=i/part.length*width,h=Math.max(1,value/max*(area-8));ctx.fillStyle=x<=head?'#397859':'#b9c1b8';ctx.fillRect(x,(area-h)/2,Math.max(.7,width/part.length*.75),h);});
  ctx.font='9px Segoe UI, sans-serif'; ctx.fillStyle='#72816f';
  for(let i=0;i<=4;i++){ctx.textAlign=i===0?'left':i===4?'right':'center';ctx.fillText(formatTime(pb.selection_duration*i/4),width*i/4+(i===0?4:i===4?-4:0),height-4);}
  if(head>=0) { ctx.fillStyle='#243b36'; ctx.fillRect(Math.max(0,Math.min(width-2,head)),0,2,area); }
}
player.addEventListener('loadedmetadata',()=>{player.classList.toggle('audio-only',!player.videoWidth); const c=currentCandidate(); if(c?.offset) player.currentTime=c.offset;});
player.addEventListener('error',()=>{if(pb&&player.getAttribute('src')) playFrom(candidate+1);});
player.addEventListener('seeked',()=>{const c=currentCandidate(); if(c&&player.currentTime<c.offset-.05) player.currentTime=c.offset;});
player.addEventListener('timeupdate',()=>{
  const c=currentCandidate(); if(!c) return;
  if(c.end!=null&&player.currentTime>=c.end&&!player.paused) player.pause();
  const t=player.currentTime-c.offset, index=pb.cues.findIndex(q=>t>=q.start&&t<q.end);
  if(index!==cueIndex) {
    const items=$('cues').children; items[cueIndex]?.classList.remove('active'); cueIndex=index;
    $('subtitle').textContent=index<0?'':pb.cues[index].text;  // untrusted text: textContent only
    if(index>=0) { items[index].classList.add('active'); $('cues').scrollTop=items[index].offsetTop-$('cues').clientHeight/2; }
  }
  drawReview();
});
$('review-wave').onclick=e=>{const c=currentCandidate(); if(c) player.currentTime=c.offset+e.offsetX/e.currentTarget.clientWidth*pb.selection_duration;};
new ResizeObserver(drawReview).observe($('review-wave'));
$('review-open').onclick=()=>task(openReview);
$('review-back').onclick=()=>task(async()=>{await releasePlayback();show(5);});
$('review-again').onclick=()=>$('again').click();
window.addEventListener('pywebviewready',()=>task(async()=>{const data=await call('queue');$('language').value=data.language;if($('language').value!==data.language){const extra=document.createElement('option');extra.value=extra.textContent=data.language;$('language').append(extra);$('language').value=data.language;}$('context').value=data.context;await renderQueue(data);show(1);}));
