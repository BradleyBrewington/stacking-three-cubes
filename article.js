(()=>{'use strict';const data=JSON.parse(document.getElementById('draftData').textContent);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=s=>s==null?'—':Math.floor(Math.round(s)/60)+':'+String(Math.round(s)%60).padStart(2,'0');
function timeLink(r,label,seconds,videoTime){return seconds==null?'—':`<button type="button" class="time-link" data-video-time="${videoTime??seconds/(r.playbackSpeed||1)}" title="Play from two video seconds before this event" aria-label="Jump to ${esc(label)} at ${fmt(seconds)} in ${r.group} trial ${r.ordinal}">${fmt(seconds)}</button>`}
function card(r){
 const metric=(label,value)=>`<div class="metric">${label}<strong>${value}</strong></div>`;
 const milestone=(label,key)=>metric(label,timeLink(r,label,r[key],r.videoSeek?.[key]));
 const events=(r.events||[]).map(e=>`<li>${e.window?e.window.map((s,i)=>timeLink(r,'placement event',s,e.videoWindow[i])).join('–'):timeLink(r,'placement event',e.time,e.videoTime)}: ${esc(e.description)}</li>`).join('');
 return `<article class="run" id="trial-${r.group}-${String(r.ordinal).padStart(2,'0')}" aria-label="${r.group} run ${r.id}"><details class="trial-details"><summary class="run-head"><h4>Trial ${String(r.ordinal).padStart(2,'0')}</h4><span class="badge ${r.outcome==='successful'?'':'failed'}">${r.outcome==='successful'?'Stacked':'Unsuccessful'}</span></summary><div class="trial-body"><div class="media"><video controls preload="none" playsinline poster="${esc(r.poster)}" aria-label="Overhead recording for ${r.group} trial ${r.ordinal}"><source src="${esc(r.video)}" type="video/mp4"></video></div><div class="video-caption">${r.playbackSpeed||1}&times; playback &middot; ${fmt(r.videoDuration)}</div><div class="metrics">${milestone('Run length','duration')}${metric('Grasp attempts',r.graspAttempts)}${milestone('First grasp','firstGrasp')}${milestone('First pickup','firstLift')}${milestone('Two cubes','firstStack')}${milestone('Three cubes','completion')}</div>${events?`<ul class="run-events">${events}</ul>`:''}${r.agentQuote?`<div class="failure-detail"><strong>${esc(r.failureCategory)}</strong><p>${esc(r.failureReason)}</p><details ${r.agentQuote.length<500?'open':''}><summary>Exact agent message</summary><blockquote class="agent-quote"><p>${esc(r.agentQuote)}</p></blockquote></details></div>`:''}<p class="run-note" data-edit="note-${r.id}" data-placeholder="Add a run note">${esc(r.notes)}</p></div></details></article>`;
}
document.getElementById('astraRuns').innerHTML=data.runs.filter(r=>r.group==='astra').map(card).join('');document.getElementById('fableRuns').innerHTML=data.runs.filter(r=>r.group==='fable').map(card).join('');
document.querySelectorAll('.trial-details').forEach(el=>el.addEventListener('toggle',()=>{if(!el.open)el.querySelector('video').pause()}));
for(const group of ['astra','fable']){const completed=data.runs.filter(r=>r.group===group&&r.outcome==='successful'&&Number.isFinite(r.completion));document.getElementById(group+'Average').textContent=completed.length?fmt(completed.reduce((sum,r)=>sum+r.completion,0)/completed.length):'?';}
function playTrialAt(article,videoTime){
 article.querySelector('.trial-details').open=true;
 const video=article.querySelector('video');
 document.querySelectorAll('video').forEach(other=>{if(other!==video)other.pause()});
 const seek=()=>{video.currentTime=Math.max(0,Math.min(videoTime-2,video.duration-.025));video.scrollIntoView({block:'center',behavior:'instant'});video.play().catch(()=>{});};
 if(video.readyState>=1)seek();else{video.addEventListener('loadedmetadata',seek,{once:true});video.load();}
}
document.getElementById('runs').addEventListener('click',e=>{
 const button=e.target.closest('[data-video-time]');if(!button)return;
 playTrialAt(button.closest('.run'),Number(button.dataset.videoTime));
});
document.getElementById('graphs').addEventListener('click',e=>{
 const link=e.target.closest('[data-stage-run]');if(!link)return;e.preventDefault();
 const run=data.runs.find(r=>r.id===Number(link.dataset.stageRun));
 const stage=Number(link.dataset.stage);const key=stage===1?'firstLift':'firstStack';
 const seconds=stage===0?0:(run.videoSeek?.[key]??run[key]/(run.playbackSpeed||1));
 playTrialAt(document.getElementById(`trial-${run.group}-${String(run.ordinal).padStart(2,'0')}`),seconds);
});

})();
