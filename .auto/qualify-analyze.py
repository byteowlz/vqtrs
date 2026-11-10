import json, statistics
from pathlib import Path
Q=Path('.auto/results/qualification')
labels=['onnx-minilm','qwen3-0.6b','sparse-splade','m3-bgem3','rerank-bge-base']
summary={}
for label in labels:
    runs=[json.loads((Q/f'{label}-{i}.json').read_text()) for i in (1,2,3,4)]
    ids=[c['id'] for c in runs[0]['cases']]
    for r in runs[1:]:
        assert [c['id'] for c in r['cases']]==ids, f'{label}: case set changed'
    changed=0; cases={}
    for k,case_id in enumerate(ids):
        outputs=[r['cases'][k]['output'] for r in runs]
        if any(o!=outputs[0] for o in outputs[1:]): changed+=1
        med=[statistics.median(r['cases'][k]['ms']) for r in runs]
        cases[case_id]={'per_run_ms':med,'median_ms':statistics.median(med),
                        'spread':max(med)/min(med),'rows':runs[0]['cases'][k]['rows']}
    totals=[sum(statistics.median(c['ms']) for c in r['cases']) for r in runs]
    summary[label]={'kind':runs[0].get('kind','dense'),'model':runs[0]['model'],
        'backend':runs[0].get('backend','onnx'),'runs':len(runs),'cases':len(ids),
        'suite_rows':sum(c['rows'] for c in runs[0]['cases']),
        'suite_ms_per_run':totals,'suite_ms_median':statistics.median(totals),
        'same_binary_spread':max(totals)/min(totals),
        'changed_output_cases_across_processes':changed,
        'worst_case_spread':max(v['spread'] for v in cases.values()),
        'load_ms_per_run':[r['load_ms'] for r in runs],
        'cases_detail':cases}
    print(f"{label:18s} rows={summary[label]['suite_rows']:3d} median_ms={statistics.median(totals):9.1f} "
          f"spread={summary[label]['same_binary_spread']:.2f}x worst_case={summary[label]['worst_case_spread']:.2f}x "
          f"changed={changed}")
Path('.auto/results/qualification/summary.json').write_text(json.dumps(summary,indent=2))
