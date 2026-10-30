"""Freeze and supervise one reviewed candidate, controls and paired transit."""
import ast
from contextlib import redirect_stdout,redirect_stderr
from dataclasses import is_dataclass,fields
from datetime import datetime,timezone
from fractions import Fraction as Q
import importlib.metadata,importlib.util
import os
from pathlib import Path
import subprocess,sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PYTHON=ROOT/'.venv313/Scripts/python.exe'
HELPER=ROOT/'artifacts/parent-child-cut-direct-response-20261023'
TEST='tests/research/test_native_capacity_selection.py'
ADAPTER='src/tnfr/research/native_capacity_selection.py'
PROOF='theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md'
for directory in (ROOT,ROOT/'src'):
    if str(directory) in sys.path:sys.path.remove(str(directory))
    sys.path.insert(0,str(directory))
from tnfr.research.artifact_io import write_json_once,file_receipt,verify_archive_members,encode_exact_tree
from tnfr.utils.io import json_loads,json_dumps

BUDGET={'watchdog_wall_budget_seconds':1200,'maximum_rss_bytes':2*1024**3,'maximum_response_bytes':64*1024**2,'maximum_log_bytes':1024**2,'diagnostic_reserve_bytes':1024**2,'handshake_timeout_seconds':30,'poll_seconds':0.1}
EXCLUDED={'evaluated-source.zip','baseline.json','before-edit.zip','validation-plan.json'}
RESOURCE='Total 1200-second watchdog includes handshake, setup, controls, seed generation and both interleaved transits; Windows kill-on-close job, sampled aggregate RSS and actual-worker OS peak, not a hard RSS allocation limit. All task-folder outputs except baseline/before-edit/source archive/validation plan count toward 64 MiB; job log has its own 1 MiB bound. Cleanup is separately bounded.'
def read(name):return json_loads((OUT/name).read_bytes())
def receipt(path):return file_receipt(path,label=Path(path).relative_to(ROOT).as_posix())
def utc():return datetime.now(timezone.utc).isoformat()
def require(condition,message):
    if not condition:raise ValueError(message)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
def inventory():return sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'src/tnfr').rglob('*.py'))
def environment():
    import tnfr
    require(Path(sys.executable).resolve()==PYTHON.resolve(),'wrong interpreter')
    require(Path(tnfr.__file__).resolve()==(ROOT/'src/tnfr/__init__.py').resolve(),'wrong TNFR source')
    return {'python':sys.executable,'version':sys.version,'tnfr_source':tnfr.__file__,'packages':{name:importlib.metadata.version(name) for name in ('pytest','numpy','scipy','sympy','networkx')}}
def detach(value):
    if is_dataclass(value):return {f.name:detach(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,dict):return {key:detach(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [detach(item) for item in value]
    return value
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
def supervisor():return module(HELPER/'supervision.py','selection_supervision')
def verify_baseline():
    before=read('baseline.json');allowed=set(before['allowed_changed_paths'])
    require(git('rev-parse','HEAD').strip()==before['head'],'HEAD changed')
    for name,digest in before['source_sha256'].items():
        if name not in allowed:require(receipt(ROOT/name)['sha256']==digest,'unrelated source changed: '+name)
    current={line[3:]:line[:2] for line in git('status','--porcelain=v1','--untracked-files=all').splitlines()}
    initial={line[3:]:line[:2] for line in before['dirty_paths_before'].splitlines()}
    require({name for name in current.keys()|initial.keys() if current.get(name)!=initial.get(name)}<=allowed,'unrelated status change')
    return before
def test_names():
    tree=ast.parse((ROOT/TEST).read_text(encoding='utf-8'))
    return [node.name for node in tree.body if isinstance(node,ast.FunctionDef) and node.name.startswith('test_')]
def pytest_args(names):return ['-q','-s','-p','no:cacheprovider','-o','addopts=','--junitxml='+str(OUT/'control-results.xml'),*[TEST+'::'+name for name in names]]
def freeze():
    before=verify_baseline();review=read('pre-evaluation-review.json');protocol=read('prospective-run.json')
    require(review['verdict']=='PASS' and review['independent'] is True and review['author']!=review['reviewer'],'independent review required')
    require(protocol['scope']==receipt(OUT/'scope.json') and protocol['independent_review']==receipt(OUT/'pre-evaluation-review.json'),'protocol associations differ')
    require(protocol['budget']==BUDGET and protocol['resource_scope']==RESOURCE,'budget mismatch')
    require(protocol['test_names']==test_names(),'test selection differs')
    require({key:value for key,value in protocol.items() if key!='independent_review'}==review['substantive_protocol'],'reviewed substantive protocol changed')
    for item in review['sources']:require(receipt(ROOT/item['path'])==item,'reviewed source changed')
    names=set(inventory())|set(before['allowed_changed_paths'])
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'tests').rglob('conftest.py'))
    names.update(['AGENTS.md','.github/agents/my-agent.md','README.md','TESTING.md','pyproject.toml','theory/nodal/RELATIONAL_DOMAIN_AND_CAPTURE.md','theory/nodal/RELATIONAL_RECOVERY_AND_INTERACTION.md'])
    names.update((OUT/name).relative_to(ROOT).as_posix() for name in ('validate.py','seed_recipe.py','scope.json','prospective-run.json','pre-evaluation-review.json'))
    names.update((HELPER/name).relative_to(ROOT).as_posix() for name in ('supervision.py','bootstrap.py'))
    sources=[receipt(ROOT/name) for name in sorted(names)]
    with ZipFile(OUT/'evaluated-source.zip','x',ZIP_DEFLATED) as stream:
        for item in sources:stream.write(ROOT/item['path'],arcname=item['path'])
    verify_archive_members(OUT/'evaluated-source.zip',{item['path']:item['sha256'] for item in sources})
    write_json_once(OUT/'validation-plan.json',{'frozen_utc':utc(),'head':before['head'],'source':sources,'source_archive':receipt(OUT/'evaluated-source.zip'),'native_python_inventory':inventory(),'environment':environment(),'protocol':receipt(OUT/'prospective-run.json'),'review':receipt(OUT/'pre-evaluation-review.json'),'test_names':protocol['test_names'],'budget':BUDGET,'resource_scope':RESOURCE})
    print({'status':'reviewed_source_frozen','sources':len(sources)},flush=True)
def verify_frozen(plan):
    require(git('rev-parse','HEAD').strip()==plan['head'],'frozen HEAD changed')
    require(inventory()==plan['native_python_inventory'] and environment()==plan['environment'],'frozen environment/inventory changed')
    require(plan['budget']==BUDGET and plan['resource_scope']==RESOURCE,'frozen budget changed')
    for item in plan['source']+[plan['source_archive'],plan['protocol'],plan['review']]:
        require(receipt(ROOT/item['path'])==item,'frozen source changed: '+item['path'])
    verify_archive_members(ROOT/plan['source_archive']['path'],{item['path']:item['sha256'] for item in plan['source']})
def worker():
    supervisor().validate_bootstrap_admission(globals().get('__tnfr_supervision__'),expected_prefix=ROOT/'.venv313',expected_executable=PYTHON)
    plan=read('validation-plan.json');verify_frozen(plan)
    require(read('run-start.json')['plan']==receipt(OUT/'validation-plan.json'),'start does not bind plan')
    os.chdir(ROOT);os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';os.environ.pop('PYTEST_ADDOPTS',None)
    os.environ.pop('TNFR_TEST_MATH_BACKEND',None);os.environ.pop('TNFR_MATH_BACKEND',None)
    with (OUT/'controls.stdout.log').open('x',encoding='utf-8') as stdout,(OUT/'controls.stderr.log').open('x',encoding='utf-8') as stderr:
        with redirect_stdout(stdout),redirect_stderr(stderr):
            import pytest
            code=int(pytest.main(pytest_args(plan['test_names'])))
    require(code==0,'pre-candidate controls failed')
    cases=ET.parse(OUT/'control-results.xml').getroot()
    require(not any(cases.findall('.//'+tag) for tag in ('failure','error','skipped')),'nonpassing controls')
    require(sorted(case.get('name') for case in cases.findall('.//testcase'))==sorted(plan['test_names']),'control inventory differs')
    write_json_once(OUT/'control-validation.json',{'passed':True,'tests':plan['test_names'],'junit':receipt(OUT/'control-results.xml')})
    candidate=module(OUT/'seed_recipe.py','selection_candidate_recipe').make_candidate()
    initial=tuple(candidate['initial'])
    require(len(initial)==6 and all(type(x) is Q for x in initial),'candidate is not exact six-coordinate state')
    require(initial[4]==initial[5]==Q(33,32),'capacity preparation differs')
    write_json_once(OUT/'candidate.json',{'created_utc':utc(),'recipe':receipt(OUT/'seed_recipe.py'),'selection':candidate,'exact_state_shared_by_both_models':True,'finite_admission':'pending paired whole-time certificate'})
    print({'status':'candidate_frozen','initial_float':[float(x) for x in initial]},flush=True)
    from tnfr.mathematics._rational_interval import I,pi_interval
    from tnfr.research.native_capacity_selection import step,capture,domain
    box=tuple(I(value) for value in initial)
    require(min(domain(box))>0,'candidate outside full regular/positive domain')
    q,r,a,b,s,t=box;pi=pi_interval()
    require(min((a-pi/2).lo,(pi-a).lo,b.lo,(pi/2-b).lo,(a-b).lo,(pi-(a-b)).lo)>0,'initial winding-one chart not certified')
    state={name:{'box':box,'steps':0,'status':'running','sector':None,'reason':None} for name in ('held','dynamic')}
    initial_capture=capture(box)
    if initial_capture['sector'] is not None:
        for name,item in state.items():
            item.update(status='captured',sector=initial_capture['sector'])
            write_json_once(OUT/(name+'-capture.json'),{'time':Q(0),'accepted_steps':0,'endpoint':detach(box),'observation':detach(initial_capture)})
    logs={name:(OUT/(name+'-steps.jsonl')).open('x',encoding='utf-8') for name in state}
    try:
        for index in range(512):
            for name,item in state.items():
                if item['status']!='running':continue
                record,failed,reason=step(item['box'],dynamic=name=='dynamic',duration=Q(1,8),order=14,time=Q(index,8))
                if record is None:
                    item.update(status='unavailable',reason=reason)
                    write_json_once(OUT/(name+'-failure.json'),{'time':Q(index,8),'reason':reason,'failed_tube':detach(failed),'last_box':detach(item['box']),'accepted_steps':item['steps']})
                    print({'model':name,'status':'unavailable','time':index/8,'reason':reason},flush=True)
                    continue
                logs[name].write(json_dumps(encode_exact_tree(detach(record)),allow_nan=False)+'\n');logs[name].flush()
                item['box']=record.endpoint;item['steps']+=1
                observation=capture(record.endpoint)
                if observation['sector'] is not None:
                    item.update(status='captured',sector=observation['sector'])
                    write_json_once(OUT/(name+'-capture.json'),{'time':Q(index+1,8),'accepted_steps':item['steps'],'endpoint':detach(record.endpoint),'observation':detach(observation)})
                    print({'model':name,'status':'captured','sector':observation['sector'],'time':(index+1)/8},flush=True)
            if (index+1)%8==0:print({'time':(index+1)/8,'models':{name:{'status':item['status'],'steps':item['steps']} for name,item in state.items()}},flush=True)
            if all(item['status']!='running' for item in state.values()):break
    finally:
        for stream in logs.values():stream.close()
    for item in state.values():
        if item['status']=='running':item.update(status='unavailable',reason='declared_horizon_exhausted')
    passed=all(item['status']=='captured' for item in state.values()) and {item['sector'] for item in state.values()}=={0,1}
    orientation=passed and state['held']['sector']==1 and state['dynamic']['sector']==0
    write_json_once(OUT/'paired-result.json',{'finished_utc':utc(),'passed':passed,'predicted_orientation_matched':orientation,'candidate':receipt(OUT/'candidate.json'),'models':detach(state),'step_logs':{name:receipt(OUT/(name+'-steps.jsonl')) for name in state},'new_candidate_count':1,'parameter_scans':0,'prior_producer_replays':0,'physical_identification':False})
    verify_frozen(plan)
    write_json_once(OUT/'worker-result.json',{'finished_utc':utc(),'exit_code':0,'paired_certificate_passed':passed,'plan':receipt(OUT/'validation-plan.json')})
    print({'status':'paired_evaluation_complete','passed':passed,'orientation_matched':orientation},flush=True)
    return 0
def run():
    plan=read('validation-plan.json');verify_baseline();verify_frozen(plan)
    for name in ('run-start.json','validation.json','validation-failure.json','candidate.json','worker-result.json','paired-result.json','control-results.xml','controls.stdout.log','controls.stderr.log'):
        require(not (OUT/name).exists(),'existing output prohibits retry: '+name)
    write_json_once(OUT/'run-start.json',{'started_utc':utc(),'attempt':1,'plan':receipt(OUT/'validation-plan.json'),'budget':BUDGET,'resource_scope':RESOURCE})
    result=None
    try:
        folder=OUT/'job';folder.mkdir(exist_ok=False)
        result=supervisor().run_supervised(interpreter=PYTHON,worker=Path(__file__),worker_args=('--worker',),folder=folder,bootstrap_path=HELPER/'bootstrap.py',expected_prefix=ROOT/'.venv313',budget=BUDGET,response_bytes=lambda:sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and p.name not in EXCLUDED))
        require(result['reason'] is None and result['worker_exit_code']==0,'supervised worker failed: '+str(result['reason']))
        require(result['cleanup_complete'] is True and result['job_empty_after_cleanup'] is True,'owned job cleanup incomplete')
        verify_frozen(plan)
        paired=read('paired-result.json')
        write_json_once(OUT/'validation.json',{'completed_utc':utc(),'evaluation_completed':True,'paired_certificate_passed':paired['passed'],'orientation_matched':paired['predicted_orientation_matched'],'plan':receipt(OUT/'validation-plan.json'),'controls':receipt(OUT/'control-validation.json'),'paired_result':receipt(OUT/'paired-result.json'),'worker_result':receipt(OUT/'worker-result.json'),'job_result':receipt(folder/'supervision-result.json'),'resource_result':result,'budget':BUDGET,'resource_scope':RESOURCE,'retry_permitted':False})
        print({'status':'evaluation_recorded','paired_certificate_passed':paired['passed'],'seconds':result['elapsed_seconds'],'peak_rss':result['peak_aggregate_working_set_bytes']},flush=True)
    except BaseException as exc:
        write_json_once(OUT/'validation-failure.json',{'failed_utc':utc(),'error':f'{type(exc).__name__}: {exc}','plan':receipt(OUT/'validation-plan.json'),'resource_result':result,'retained_outputs':[receipt(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in EXCLUDED],'retry_permitted':False})
        raise
if __name__=='__main__':
    require(len(sys.argv)==2,'select --freeze, --run or supervised --worker')
    if sys.argv[1]=='--freeze':freeze()
    elif sys.argv[1]=='--run':run()
    elif sys.argv[1]=='--worker':raise SystemExit(worker())
    else:raise ValueError('unknown command')
