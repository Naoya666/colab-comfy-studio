import { apiFetch } from './gpu-api';
export interface LibraryModel {
  id: string; display_name: string; filename: string; model_type: string; architecture: string;
  state: string; size?: number; sha256?: string; expected_sha256?: string; error?: string;
  tags?: string[]; trigger_words?: string[]; drive_file_id?: string; source?: string; version_name?: string;
}
export interface ModelTask { id: string; model_id: string; state: string; bytes: number; total?: number; speed?: number; error?: string; }
export interface Dependency {
  id: string; node_id: string; node_type: string; field: string; reference: string; model_type: string;
  architecture: string; model_id?: string; status: string; required: boolean; disabled?: boolean;
  strengths?: Record<string, number>; candidates?: string[];
}
export interface DependencyRecord { dependencies: Dependency[]; models_ready: boolean; workflow_state: string; runtime_validated: boolean; }
export interface Catalog { items: LibraryModel[]; types: Record<string, string>; statistics: { count: number; bytes: number; by_type: Record<string, {count: number; bytes: number}> }; }
export const libraryApi = <T,>(path = '', body?: unknown, method = body === undefined ? 'GET' : 'POST') =>
  apiFetch<T>(`/api/gpu/model-library${path}`, { method, ...(body === undefined ? {} : {body: JSON.stringify(body)}) });
export const dependencyApi = <T,>(workflow: string, action = '', body?: unknown) =>
  apiFetch<T>(`/api/gpu/workflows/${encodeURIComponent(workflow)}/model-dependencies${action ? '/' + action : ''}`, body === undefined && !action ? {} : {method: 'POST', body: JSON.stringify(body || {})});
export const bytesLabel = (value?: number) => value ? `${(value / 1024 ** 3).toFixed(2)} GB` : '大小待确认';
export const modelStatus: Record<string, string> = {pending:'待准备', resolving:'解析中', downloading:'下载中', verifying:'校验中', uploading:'写入 Drive', ready:'Drive 已就绪', failed:'失败', missing:'Drive 文件缺失', unverified:'待校验', cancelled:'已取消', matched:'已绑定', unresolved:'待补充来源', needs_confirmation:'候选待确认', manual_review:'节点待人工确认'};

// Drive resumable sessions are bearer capabilities. Keep them out of localStorage and logs.
export async function uploadModel(file: File, onProgress: (sent: number) => void) {
  const session = await libraryApi<{upload_url: string; chunk_size: number}>('/upload-session', {filename: file.name, size:file.size});
  let offset = 0;
  let result: {id?: string} = {};
  while (offset < file.size) {
    const end = Math.min(offset + session.chunk_size, file.size);
    let completed = false;
    for (let attempt = 0; attempt < 3 && !completed; attempt++) {
      try {
        const response = await fetch(session.upload_url, {method:'PUT', body:file.slice(offset, end), headers:{'Content-Range':`bytes ${offset}-${end - 1}/${file.size}`, 'Content-Type':'application/octet-stream'}});
        if (response.status === 308) {
          const range = response.headers.get('Range');
          offset = range ? Number(range.match(/-(\d+)$/)?.[1] ?? -1) + 1 : end;
          completed = true;
        } else if (response.ok) { result = await response.json(); offset = end; completed = true; }
        else throw new Error(`Drive 上传失败（HTTP ${response.status}）`);
      } catch (error) {
        if (attempt === 2) throw error;
        const probe = await fetch(session.upload_url, {method:'PUT', headers:{'Content-Range':`bytes */${file.size}`}});
        if (probe.ok) { result = await probe.json(); offset = file.size; completed = true; }
        else if (probe.status === 308) { const range = probe.headers.get('Range'); offset = range ? Number(range.match(/-(\d+)$/)?.[1] ?? -1) + 1 : 0; completed = true; }
        else throw new Error(`Drive 上传会话失效（HTTP ${probe.status}），请重新上传`);
      }
    }
    onProgress(offset);
  }
  if (!result.id) throw new Error('Drive 未返回文件 ID，不能提交模型索引');
  return result.id;
}
