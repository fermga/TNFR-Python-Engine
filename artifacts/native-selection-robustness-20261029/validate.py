"""Freeze and supervise one static audit; never rerun a trajectory producer."""
from datetime import datetime,timezone
import importlib.util,importlib.metadata
from pathlib import Path
import subprocess,sys
from zipfile import ZipFile,ZIP_DEFLATED
from tnfr.research.artifact_io import file_receipt,write_json_once,verify_archive_members
from tnfr.utils.io import json_loads

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PYTHON=ROOT/'.venv313/Scripts/python.exe'
HELPER=ROOT/'artifacts/parent-child-cut-direct-response-20261023'
BUDGET={'watchdog_wall_budget_seconds':180,'maximum_rss_bytes':2147483648,'maximum_response_bytes':16777216,'maximum_log_bytes':1048576,'diagnostic_reserve_bytes':1048576,'handshake_timeout_seconds':30,'poll_seconds':0.1}
RESOURCE='One static audit of existing primitive boxes and source associations; 180-second supervised watchdog includes interpreter handshake, verification and audit; root preflight/freeze metadata checks are outside this worker allowance, 2 GiB sampled aggregate RSS (not a hard allocation cap), 16 MiB task outputs excluding baseline/before-edit/evaluated-source/validation-plan. Job log bound1MiB; cleanup separately bounded. No trajectory, candidate generator or historical producer runs.'
EXCLUDED={'baseline.json','before-edit.zip','evaluated-source.zip','validation-plan.json'}
def require(value,message):
    if not value:raise ValueError(message)
def read(name):return json_loads((OUT/name).read_bytes())
def receipt(path):return file_receipt(path,label=path.relative_to(ROOT).as_posix())
def utc():return datetime.now(timezone.utc).isoformat()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
def inventory():return sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'src/tnfr').rglob('*.py'))
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result);return result
def environment():
    import tnfr
    require(Path(sys.executable).resolve()==PYTHON.resolve(),'wrong interpreter')
    require(Path(tnfr.__file__).resolve()==ROOT/'src/tnfr/__init__.py','wrong source')
    return {'python':sys.executable,'version':sys.version,'tnfr_source':tnfr.__file__,'packages':{name:importlib.metadata.version(name) for name in ('numpy','scipy','networkx')}}
def supervisor():return module(HELPER/'supervision.py','robust_supervision')
def baseline():
    value=read('baseline.json');require(git('rev-parse','HEAD').strip()==value['head'],'HEAD changed')
    for name,digest in value['source_sha256'].items():require(receipt(ROOT/name)['sha256']==digest,'pre-evaluation source changed: '+name)
    require(git('status','--porcelain=v1','--untracked-files=all')==value['dirty_paths_before'],'pre-evaluation status changed')
    return value
def freeze():
    before=baseline();protocol=read('prospective-run.json');review=read('pre-evaluation-review.json')
    require(review['verdict']=='PASS' and review['independent'] is True and review['author']!=review['reviewer'],'independent review absent')
    require(protocol['independent_review']==receipt(OUT/'pre-evaluation-review.json'),'review association changed')
    require({k:v for k,v in protocol.items() if k!='independent_review'}==review['substantive_protocol'],'protocol differs from review')
    require(protocol['budget']==BUDGET and protocol['resource_scope']==RESOURCE,'resource declaration differs')
    for item in review['sources']:require(receipt(ROOT/item['path'])==item,'reviewed source changed')
    names=set(inventory())|set(before['allowed_changed_paths'])
    names.update(['AGENTS.md','.github/agents/my-agent.md','README.md','TESTING.md','pyproject.toml','scripts/check_research_prerequisites.py','theory/nodal/RELATIONAL_RECOVERY_AND_INTERACTION.md','theory/nodal/RELATIONAL_DOMAIN_AND_CAPTURE.md'])
    names.update((OUT/name).relative_to(ROOT).as_posix() for name in ('audit.py','validate.py','scope.json','robustness-section.md','prospective-run.json','pre-evaluation-review.json'))
    names.update((HELPER/name).relative_to(ROOT).as_posix() for name in ('supervision.py','bootstrap.py'))
    sources=[receipt(ROOT/name) for name in sorted(names)]
    with ZipFile(OUT/'evaluated-source.zip','x',ZIP_DEFLATED) as archive:
        for item in sources:archive.write(ROOT/item['path'],arcname=item['path'])
    verify_archive_members(OUT/'evaluated-source.zip',{i['path']:i['sha256'] for i in sources})
    write_json_once(OUT/'validation-plan.json',{'frozen_utc':utc(),'head':before['head'],'source':sources,'source_archive':receipt(OUT/'evaluated-source.zip'),'inventory':inventory(),'environment':environment(),'protocol':receipt(OUT/'prospective-run.json'),'review':receipt(OUT/'pre-evaluation-review.json'),'budget':BUDGET,'resource_scope':RESOURCE,'inputs':protocol['inputs']})
    print({'status':'static_source_frozen','sources':len(sources)},flush=True)
def verify(plan):
    require(git('rev-parse','HEAD').strip()==plan['head'],'HEAD differs')
    require(environment()==plan['environment'] and inventory()==plan['inventory'],'environment/inventory differs')
    for item in plan['source']+plan['inputs']+[plan['source_archive'],plan['protocol'],plan['review']]:require(receipt(ROOT/item['path'])==item,'frozen source/input changed: '+item['path'])
    verify_archive_members(ROOT/plan['source_archive']['path'],{i['path']:i['sha256'] for i in plan['source']})
def worker():
    supervisor().validate_bootstrap_admission(globals().get('__tnfr_supervision__'),expected_prefix=ROOT/'.venv313',expected_executable=PYTHON)
    plan=read('validation-plan.json');verify(plan)
    require(read('run-start.json')['plan']==receipt(OUT/'validation-plan.json'),'run not bound to plan')
    guard=subprocess.run([str(PYTHON),'scripts/check_research_prerequisites.py'],cwd=ROOT,capture_output=True,text=True,timeout=30)
    write_json_once(OUT/'pre-audit-prerequisites.json',{'exit_code':guard.returncode,'stdout':guard.stdout,'stderr':guard.stderr,'registry':receipt(ROOT/'theory/research_prerequisites.json')})
    require(guard.returncode==0,'prerequisite integrity failure')
    result=module(OUT/'audit.py','robust_static_audit').audit()
    write_json_once(OUT/'static-audit-result.json',result)
    require(result['passed'] is True,'retained evidence does not satisfy analytical admission')
    verify(plan)
    write_json_once(OUT/'worker-result.json',{'finished_utc':utc(),'passed':True,'plan':receipt(OUT/'validation-plan.json'),'static_audit':receipt(OUT/'static-audit-result.json'),'new_trajectories':0,'historical_producer_replays':0})
    print({'status':'static_audit_passed','new_trajectories':0},flush=True)
def run():
    plan=read('validation-plan.json');baseline();verify(plan)
    for name in ('run-start.json','validation.json','validation-failure.json','static-audit-result.json','worker-result.json'):require(not (OUT/name).exists(),'existing output prohibits retry: '+name)
    write_json_once(OUT/'run-start.json',{'started_utc':utc(),'attempt':1,'plan':receipt(OUT/'validation-plan.json'),'budget':BUDGET,'resource_scope':RESOURCE})
    resources=None
    try:
        folder=OUT/'job';folder.mkdir(exist_ok=False)
        resources=supervisor().run_supervised(interpreter=PYTHON,worker=Path(__file__),worker_args=('--worker',),folder=folder,bootstrap_path=HELPER/'bootstrap.py',expected_prefix=ROOT/'.venv313',budget=BUDGET,response_bytes=lambda:sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and p.name not in EXCLUDED))
        require(resources['reason'] is None and resources['worker_exit_code']==0,'supervised audit failed: '+str(resources['reason']))
        require(resources['cleanup_complete'] and resources['job_empty_after_cleanup'],'cleanup incomplete')
        verify(plan)
        write_json_once(OUT/'validation.json',{'completed_utc':utc(),'passed':True,'evaluation_kind':'static exact audit of retained evidence','plan':receipt(OUT/'validation-plan.json'),'audit':receipt(OUT/'static-audit-result.json'),'worker':receipt(OUT/'worker-result.json'),'resource_result':resources,'budget':BUDGET,'resource_scope':RESOURCE,'new_trajectories':0,'historical_producer_replays':0,'retry_permitted':False})
        print({'status':'static_audit_validated','seconds':resources['elapsed_seconds'],'peak_rss':resources['peak_aggregate_working_set_bytes']},flush=True)
    except BaseException as exc:
        write_json_once(OUT/'validation-failure.json',{'failed_utc':utc(),'error':f'{type(exc).__name__}: {exc}','plan':receipt(OUT/'validation-plan.json'),'resource_result':resources,'retry_permitted':False})
        raise
if __name__=='__main__':
    require(len(sys.argv)==2,'select --freeze or --run')
    if sys.argv[1]=='--freeze':freeze()
    elif sys.argv[1]=='--run':run()
    elif sys.argv[1]=='--worker':worker()
    else:raise ValueError('unknown command')
