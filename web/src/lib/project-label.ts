export type ProjectOption = { id: string; title?: string; is_default?: boolean; system?: boolean };

export function projectOptionLabel(project: ProjectOption, projects: ProjectOption[]): string {
  const title = (project.title || '').trim() || '未命名项目';
  const duplicates = projects.filter((item) => ((item.title || '').trim() || '未命名项目') === title).length;
  return duplicates > 1 ? `${title} · ${project.id.slice(0, 6)}` : title;
}

/** Real activation uses the project the user already selected. It never picks another row. */
export function selectActivationProjectId(projects: ProjectOption[], selectedId: string): string {
  if (!selectedId) return '';
  return projects.some((project) => project.id === selectedId) ? selectedId : '';
}

export function pickInitialProjectId(projects: ProjectOption[], savedId: string): string {
  if (savedId && projects.some((p) => p.id === savedId)) return savedId;
  const preferred = projects.find((p) => p.is_default);
  if (preferred) return preferred.id;
  return projects[0]?.id || '';
}

export function projectTitleById(projectId: string | null | undefined, projects: ProjectOption[]): string {
  if (!projectId) return '';
  const match = projects.find((p) => p.id === projectId);
  if (match) return projectOptionLabel(match, projects);
  if (projectId.startsWith('native-comfy-')) return '原生 ComfyUI（系统）';
  if (projectId.startsWith('system-smoke-')) return 'GPU 可行性检查（系统）';
  return `历史项目 · ${projectId.slice(0, 6)}`;
}

export function isSystemJob(job: { system?: boolean; workflow_key?: string }): boolean {
  if (job.system === true) return true;
  return String(job.workflow_key || '').startsWith('system/');
}
