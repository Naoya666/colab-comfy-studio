import { useMemo, useRef, useState } from 'react';
import { Dices, Shuffle, Upload, X, Check } from 'lucide-react';
import { assetUrl, type GalleryAsset, type GpuManifest, type GpuParameter } from '../lib/gpu-api';
import { friendlyHint, friendlyLabel, MODE_HELP, modeHelp } from '../lib/labels';
import { InfoTip } from './ui';
import styles from '../gpu.module.css';

export type ParamSection = 'assets' | 'options' | 'advanced';

type Props = {
  manifest?: GpuManifest | null;
  values: Record<string, unknown>;
  assets: GalleryAsset[];
  uploadProgress: Record<string, number>;
  onChange: (key: string, value: unknown) => void;
  onUpload: (file: File) => Promise<GalleryAsset | void>;
  /** 已由别处处理的字段（提示词、宽高），这里不再渲染，避免重复。 */
  excludeKeys?: string[];
  /** assets=素材文件；options=常用选项（反向提示词/模式/时长等，直接展示）；advanced=其余全部（默认折叠）。 */
  section: ParamSection;
};

type Entry = { param: GpuParameter; key: string };

export function parametersOf(manifest?: GpuManifest | null): GpuParameter[] {
  const raw = manifest?.parameters;
  if (Array.isArray(raw)) return raw;
  if (!raw || typeof raw !== 'object') return [];
  return Object.entries(raw).map(([key, parameter]) => {
    const normalized = parameter as GpuParameter;
    return { ...normalized, key, id: normalized.id || key };
  });
}

function keyOf(parameter: GpuParameter, index: number) {
  return parameter.key || parameter.id || parameter.name || `parameter_${index + 1}`;
}

function optionOf(option: unknown): { value: string; label: string } {
  if (typeof option === 'object' && option !== null && 'value' in option) {
    return {
      value: String(option.value),
      label: 'label' in option && option.label ? String(option.label) : String(option.value),
    };
  }
  return { value: String(option), label: String(option) };
}

function initialValue(parameter: GpuParameter): unknown {
  if (parameter.default !== undefined) return parameter.default;
  if (parameter.widget === 'checkbox' || parameter.type === 'boolean') return false;
  if (parameter.type === 'integer' || parameter.type === 'number' || parameter.widget === 'number') return parameter.min ?? 0;
  return '';
}

function widgetOf(parameter: GpuParameter): string {
  const hasOptions = Boolean(parameter.options?.length);
  return (
    parameter.widget ||
    (parameter.type === 'text' ? 'textarea'
      : parameter.type === 'boolean' ? 'checkbox'
      : parameter.type === 'integer' || parameter.type === 'number' ? 'number'
      : ['image', 'video', 'file'].includes(parameter.type) ? 'file'
      : hasOptions ? 'select'
      : 'text')
  );
}

const TECH_KEYS = /(^|_)(steps?|seed|cfg|cfg_scale|sampler|sampler_name|scheduler|denoise|clip_skip|strength|weight)(_|$)/i;
const COMMON_KEYS = /(^|_)(mode|duration|seconds|length|quality|fps)(_|$)/i;
const NEGATIVE_PROMPT_KEY = /(^|_)negative(_prompt)?$/i;
const SEED_KEY = /(^|_)seed$/i;

/** 把参数分成三层：素材、常用选项（含反向提示词）、高级。 */
export function splitParams(manifest: GpuManifest | null | undefined, excludeKeys: string[] = []) {
  const assets: Entry[] = [];
  const options: Entry[] = [];
  const advanced: Entry[] = [];
  parametersOf(manifest).forEach((param, index) => {
    const key = keyOf(param, index);
    if (excludeKeys.includes(key)) return;
    const widget = widgetOf(param);
    const flaggedAdvanced = Boolean(param.advanced) || param.group === 'advanced';
    if (widget === 'file') assets.push({ param, key });
    else if (NEGATIVE_PROMPT_KEY.test(key)) options.push({ param, key });
    else if (!flaggedAdvanced && !TECH_KEYS.test(key) && COMMON_KEYS.test(key) && (widget === 'select' || widget === 'number')) options.push({ param, key });
    else advanced.push({ param, key });
  });
  return { assets, options, advanced };
}

export function GpuParamForm({ manifest, values, assets, uploadProgress, onChange, onUpload, excludeKeys = [], section }: Props) {
  const inputRefs = useRef<Record<string, HTMLInputElement | null>>({});
  const [openPickerKey, setOpenPickerKey] = useState<string | null>(null);
  const [dragKey, setDragKey] = useState<string | null>(null);
  const entries = useMemo(() => splitParams(manifest, excludeKeys)[section], [manifest, excludeKeys, section]);
  if (!entries.length) return null;

  function renderField({ param: parameter, key }: Entry) {
    const label = friendlyLabel(key, parameter.label || parameter.name || key);
    const current = values[key] === undefined ? initialValue(parameter) : values[key];
    const options = (parameter.options || []).map(optionOf);
    const widget = widgetOf(parameter);
    const hasRange = parameter.min !== undefined || parameter.max !== undefined;
    const hint = parameter.description || friendlyHint(key);
    const isMode = /(^|_)mode$/i.test(key);
    const isSeed = SEED_KEY.test(key);
    const isNegative = NEGATIVE_PROMPT_KEY.test(key);
    const selectedHelp = isMode ? modeHelp(current) : undefined;

    const matchingAssets = widget === 'file'
      ? assets.filter((asset) => !parameter.type || parameter.type === 'file' || asset.mime.startsWith(`${parameter.type}/`))
      : [];
    const selectedAsset = widget === 'file' ? matchingAssets.find((a) => a.id === String(current ?? '')) : undefined;

    async function handleFileUpload(file: File) {
      const uploaded = await onUpload(file);
      if (uploaded && uploaded.id) {
        onChange(key, uploaded.id);
      }
    }

    return (
      <div className={styles.paramField} key={key}>
        <div className={styles.paramLabel}>
          <label htmlFor={`gpu-${key}`}>
            {label}
            {parameter.required && <b className={styles.requiredMark} aria-label="必填">*</b>}
          </label>
          <span className={styles.paramLabelRight}>
            {hasRange && <small className="muted">范围 {parameter.min ?? 0} ~ {parameter.max ?? '∞'}</small>}
            {isMode && options.some((o) => MODE_HELP[o.value.toLowerCase()]) && (
              <InfoTip
                label={`${label}说明`}
                text={<>{options.filter((o) => MODE_HELP[o.value.toLowerCase()]).map((o) => (
                  <span key={o.value} className={styles.tipLine}><b>{MODE_HELP[o.value.toLowerCase()].name}</b>：{MODE_HELP[o.value.toLowerCase()].text}</span>
                ))}</>}
              />
            )}
          </span>
        </div>

        {widget === 'textarea' || isNegative ? (
          <textarea
            id={`gpu-${key}`}
            value={String(current ?? '')}
            rows={2}
            placeholder={isNegative ? '不希望画面中出现的元素（选填）' : undefined}
            onChange={(e) => onChange(key, e.target.value)}
          />
        ) : widget === 'select' ? (
          <select id={`gpu-${key}`} value={String(current ?? '')} onChange={(e) => onChange(key, e.target.value)}>
            {!options.some((o) => o.value === String(current ?? '')) && <option value="">请选择</option>}
            {options.map((option) => (
              <option key={option.value} value={option.value}>{modeHelp(option.value) && isMode ? `${modeHelp(option.value)!.name}（${option.value}）` : option.label}</option>
            ))}
          </select>
        ) : widget === 'checkbox' ? (
          <label className={styles.checkRow}>
            <input id={`gpu-${key}`} type="checkbox" checked={Boolean(current)} onChange={(e) => onChange(key, e.target.checked)} />
            <span>开启</span>
          </label>
        ) : widget === 'number' ? (
          isSeed ? (
            <div className={styles.seedRow}>
              <input
                id={`gpu-${key}`}
                type="number"
                inputMode="numeric"
                value={String(current ?? '')}
                min={parameter.min}
                max={parameter.max}
                step={parameter.step || 1}
                onChange={(e) => onChange(key, e.target.value === '' ? '' : Number.parseInt(e.target.value, 10))}
              />
              <button
                type="button"
                className="button secondary small"
                title="设为随机种子（-1）"
                onClick={() => onChange(key, parameter.min !== undefined && parameter.min >= 0 ? Math.floor(Math.random() * 2_000_000_000) : -1)}
              >
                <Shuffle size={14} aria-hidden />随机
              </button>
              <button
                type="button"
                className="button secondary small"
                title="生成一个固定随机数方便复现"
                onClick={() => onChange(key, Math.floor(Math.random() * 2_000_000_000))}
              >
                <Dices size={14} aria-hidden />抽固定值
              </button>
            </div>
          ) : (
            <input
              id={`gpu-${key}`}
              type="number"
              inputMode="decimal"
              value={String(current ?? '')}
              min={parameter.min}
              max={parameter.max}
              step={parameter.step || (parameter.type === 'integer' ? 1 : 'any')}
              onChange={(e) => onChange(key, e.target.value === '' ? '' : parameter.type === 'integer' ? Number.parseInt(e.target.value, 10) : Number(e.target.value))}
            />
          )
        ) : widget === 'file' ? (
          <div
            className={`${styles.fileSlot} ${dragKey === key ? styles.dropActive : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragKey(key); }}
            onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragKey(null); }}
            onDrop={(e) => {
              e.preventDefault();
              setDragKey(null);
              const file = e.dataTransfer.files?.[0];
              if (file) void handleFileUpload(file);
            }}
          >
            <div className={styles.fileControl}>
              {selectedAsset && (
                <span className={styles.filePreviewThumb}>
                  {selectedAsset.mime.startsWith('image/')
                    ? <img src={assetUrl(selectedAsset, true)} alt={selectedAsset.filename || ''} />
                    : <span>{selectedAsset.mime.startsWith('video/') ? '视频' : '文件'}</span>}
                </span>
              )}
              <select id={`gpu-${key}`} value={String(current ?? '')} onChange={(e) => onChange(key, e.target.value)}>
                <option value="">从素材库选择或拖入文件…</option>
                {matchingAssets.map((asset) => <option key={asset.id} value={asset.id}>{asset.filename || asset.id}</option>)}
              </select>
              {selectedAsset && (
                <button type="button" className={styles.iconGhost} aria-label="清除已选素材" title="清除" onClick={() => onChange(key, '')}>
                  <X size={15} aria-hidden />
                </button>
              )}
              {matchingAssets.length > 0 && (
                <button
                  type="button"
                  className="button secondary small"
                  aria-expanded={openPickerKey === key}
                  onClick={() => setOpenPickerKey(openPickerKey === key ? null : key)}
                >
                  {openPickerKey === key ? '收起图库' : '挑图'}
                </button>
              )}
              <button type="button" className="button secondary small" onClick={() => inputRefs.current[key]?.click()}>
                <Upload size={14} aria-hidden />上传
              </button>
              <input
                ref={(node) => { inputRefs.current[key] = node; }}
                type="file"
                hidden
                accept={parameter.type === 'image' ? 'image/*' : parameter.type === 'video' ? 'video/*' : undefined}
                onChange={async (event) => {
                  const file = event.target.files?.[0];
                  if (file) await handleFileUpload(file);
                  event.target.value = '';
                }}
              />
            </div>

            {openPickerKey === key && matchingAssets.length > 0 && (
              <div className={styles.refGrid}>
                {matchingAssets.map((asset) => {
                  const active = asset.id === String(current ?? '');
                  return (
                    <button
                      type="button"
                      key={asset.id}
                      className={styles.refItem}
                      data-on={active}
                      title={asset.filename || asset.id}
                      onClick={() => {
                        onChange(key, active ? '' : asset.id);
                        setOpenPickerKey(null);
                      }}
                    >
                      {asset.mime.startsWith('image/')
                        ? <img src={assetUrl(asset, true)} alt="" loading="lazy" />
                        : <span className={styles.refMedia}>{asset.mime.startsWith('video/') ? '视频' : '音频'}</span>}
                      {active && <span className={styles.refCheck}><Check size={12} strokeWidth={2.4} aria-hidden /></span>}
                    </button>
                  );
                })}
              </div>
            )}

            {Object.entries(uploadProgress).map(([name, progress]) => (
              <small className={styles.hint} key={name}>正在上传 {name}：{progress}%</small>
            ))}
          </div>
        ) : (
          <input id={`gpu-${key}`} type="text" value={String(current ?? '')} onChange={(e) => onChange(key, e.target.value)} />
        )}

        {selectedHelp ? <span className={styles.hint}>{selectedHelp.text}</span> : hint ? <span className={styles.hint}>{hint}</span> : null}
      </div>
    );
  }

  return <div className={styles.paramList}>{entries.map(renderField)}</div>;
}
