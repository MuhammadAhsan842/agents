import {readdirSync,readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
for(const dir of ['server','extension','scripts','tests'])for(const file of readdirSync(dir))if(file.endsWith('.js'))execFileSync(process.execPath,['--check',dir+'/'+file]);
const manifest=JSON.parse(readFileSync('extension/manifest.json'));
for(const file of [manifest.side_panel.default_path,manifest.background.service_worker])readFileSync('extension/'+file);
console.log('JavaScript syntax and extension entrypoints verified.');
