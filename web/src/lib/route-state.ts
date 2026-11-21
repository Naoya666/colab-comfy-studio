export type MainTab = '创作' | '任务' | '作品' | '资产' | '原生 ComfyUI' | '算力' | '设置';
export type AssetsSubTab = 'workflows' | 'models';

export interface RouteState {
  tab: MainTab;
  assetsSubTab: AssetsSubTab;
  creationApiOpen: boolean;
  jobId: string;
  assetId: string;
}

const SLUG_TO_TAB: Record<string, MainTab> = {
  create: '创作',
  tasks: '任务',
  jobs: '任务',
  gallery: '作品',
  assets: '资产',
  workflows: '资产',
  models: '资产',
  native: '原生 ComfyUI',
  compute: '算力',
  settings: '设置',
};

const TAB_TO_SLUG: Record<MainTab, string> = {
  创作: 'create',
  任务: 'tasks',
  作品: 'gallery',
  资产: 'assets',
  '原生 ComfyUI': 'native',
  算力: 'compute',
  设置: 'settings',
};

export function parseHashRoute(hash: string): RouteState {
  const cleaned = (hash || '').replace(/^#\/?/, '');
  const [rawPath, rawQuery = ''] = cleaned.split('?');
  const [slug = '', sub = ''] = rawPath.split('/').filter(Boolean);
  const params = new URLSearchParams(rawQuery);

  const tab: MainTab = SLUG_TO_TAB[slug] || '创作';
  let assetsSubTab: AssetsSubTab = 'workflows';
  if (slug === 'models' || (tab === '资产' && (sub === 'models' || params.get('sub') === 'models'))) {
    assetsSubTab = 'models';
  }
  const creationApiOpen = tab === '创作' && params.get('api') === '1';
  const jobId = tab === '任务' ? params.get('job') || '' : '';
  const assetId = tab === '作品' ? params.get('asset') || '' : '';

  return { tab, assetsSubTab, creationApiOpen, jobId, assetId };
}

export function formatHashRoute(state: RouteState): string {
  const slug = TAB_TO_SLUG[state.tab] || 'create';
  const params = new URLSearchParams();
  let path = slug;
  if (state.tab === '资产' && state.assetsSubTab === 'models') {
    path = 'assets/models';
  }
  if (state.tab === '创作' && state.creationApiOpen) {
    params.set('api', '1');
  }
  if (state.tab === '任务' && state.jobId) {
    params.set('job', state.jobId);
  }
  if (state.tab === '作品' && state.assetId) {
    params.set('asset', state.assetId);
  }
  const query = params.toString();
  return `#/${path}${query ? `?${query}` : ''}`;
}
