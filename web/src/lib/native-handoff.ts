/** Refresh the current SPA document's policy, including documents opened before an upgrade. */
export function submitNativeHandoff(doc: Document, handoff: { url: string; ticket: string }): void {
  let policy = doc.querySelector<HTMLMetaElement>('meta[name="referrer"]');
  if (!policy) {
    policy = doc.createElement('meta');
    policy.name = 'referrer';
    doc.head.append(policy);
  }
  policy.content = 'strict-origin-when-cross-origin';
  const form = doc.createElement('form');
  form.method = 'POST';
  form.action = handoff.url;
  const input = doc.createElement('input');
  input.type = 'hidden'; input.name = 'ticket'; input.value = handoff.ticket;
  form.append(input); doc.body.append(form);
  form.submit(); form.remove();
}
