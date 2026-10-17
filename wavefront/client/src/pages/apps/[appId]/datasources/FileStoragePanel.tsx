import floConsoleService from '@app/api';
import DeleteConfirmationDialog from '@app/components/DeleteConfirmationDialog';
import { Button } from '@app/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@app/components/ui/dialog';
import { Input } from '@app/components/ui/input';
import { Label } from '@app/components/ui/label';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@app/components/ui/table';
import { copyToClipboard, downloadBlobFile, extractErrorMessage } from '@app/lib/utils';
import { useNotifyStore } from '@app/store';
import { DatasourceFileEntry } from '@app/types/datasource';
import { Copy, Download, RefreshCw, Trash2, Upload } from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';

interface FileStoragePanelProps {
  datasourceId: string;
}

type PreviewKind = 'image' | 'video' | 'text' | 'unsupported';

type FilePreviewState = {
  entry: DatasourceFileEntry;
  kind: PreviewKind;
  objectUrl?: string;
  text?: string;
  contentType: string;
  sizeBytes: number;
};

const TEXT_PREVIEW_MAX_BYTES = 512 * 1024;

/** Blob responses hide the JSON error body; read it back out for the message. */
const getDownloadErrorMessage = async (error: unknown): Promise<string | undefined> => {
  const data = (error as { response?: { data?: unknown } })?.response?.data;
  if (data instanceof Blob) {
    try {
      const body = JSON.parse(await data.text()) as { meta?: { error?: string } };
      return body.meta?.error;
    } catch {
      return undefined;
    }
  }
  return extractErrorMessage(error);
};

const mimeTypeOf = (contentType: string | undefined, blobType: string) =>
  (contentType || blobType || 'application/octet-stream').split(';')[0].trim().toLowerCase();

const previewKindFor = (mime: string): PreviewKind => {
  if (mime.startsWith('image/') && mime !== 'image/svg+xml') return 'image';
  if (mime.startsWith('video/')) return 'video';
  if (
    mime.startsWith('text/') ||
    mime === 'application/json' ||
    mime === 'application/xml' ||
    mime === 'application/javascript' ||
    mime.endsWith('+json') ||
    mime.endsWith('+xml')
  ) {
    return 'text';
  }
  return 'unsupported';
};

const FileStoragePanel: React.FC<FileStoragePanelProps> = ({ datasourceId }) => {
  const { notifySuccess, notifyError } = useNotifyStore();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const previewObjectUrlRef = useRef<string | null>(null);

  const [files, setFiles] = useState<DatasourceFileEntry[]>([]);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [hasNextPage, setHasNextPage] = useState(false);
  const [pageNumber, setPageNumber] = useState(1);
  const pageSize = 50;

  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadPath, setUploadPath] = useState('');
  const [uploading, setUploading] = useState(false);

  const [downloadingPath, setDownloadingPath] = useState<string | null>(null);
  const [filePendingDelete, setFilePendingDelete] = useState<DatasourceFileEntry | null>(null);
  const [deleting, setDeleting] = useState(false);

  const [preview, setPreview] = useState<FilePreviewState | null>(null);
  const [previewLoadingPath, setPreviewLoadingPath] = useState<string | null>(null);

  const loadFiles = useCallback(
    async (page = 1) => {
      setLoadingFiles(true);
      try {
        const response = await floConsoleService.datasourcesService.listFiles(datasourceId, {
          page_number: page,
          page_size: pageSize,
        });
        const data = response.data?.data;
        setFiles(data?.files ?? []);
        setHasNextPage(Boolean(data?.has_next_page));
        setPageNumber(data?.page_number ?? page);
      } catch (error) {
        notifyError(extractErrorMessage(error) || 'Failed to list files');
        setFiles([]);
        setHasNextPage(false);
      } finally {
        setLoadingFiles(false);
      }
    },
    [datasourceId, notifyError]
  );

  useEffect(() => {
    void loadFiles(1);
  }, [loadFiles]);

  const clearPreview = useCallback(() => {
    if (previewObjectUrlRef.current) {
      URL.revokeObjectURL(previewObjectUrlRef.current);
      previewObjectUrlRef.current = null;
    }
    setPreview(null);
    setPreviewLoadingPath(null);
  }, []);

  useEffect(() => () => clearPreview(), [clearPreview]);

  const resetUploadForm = () => {
    setFile(null);
    setUploadPath('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    try {
      const response = await floConsoleService.datasourcesService.uploadFile(
        datasourceId,
        file,
        uploadPath.trim() || undefined
      );
      const path = response.data?.data?.path;
      if (!path) {
        notifyError('Upload succeeded but no path was returned');
        return;
      }
      notifySuccess('File uploaded successfully');
      setUploadDialogOpen(false);
      resetUploadForm();
      await loadFiles(1);
    } catch (error) {
      notifyError(extractErrorMessage(error) || 'Failed to upload file');
    } finally {
      setUploading(false);
    }
  };

  const handleDownload = async (entry: DatasourceFileEntry) => {
    setDownloadingPath(entry.path);
    try {
      const response = await floConsoleService.datasourcesService.downloadFile(datasourceId, entry.relative_path);
      downloadBlobFile(entry.name || 'download', response.data);
    } catch (error) {
      notifyError((await getDownloadErrorMessage(error)) || 'Failed to download file');
    } finally {
      setDownloadingPath(null);
    }
  };

  const handlePreview = async (entry: DatasourceFileEntry) => {
    clearPreview();
    setPreviewLoadingPath(entry.path);
    try {
      const response = await floConsoleService.datasourcesService.downloadFile(datasourceId, entry.relative_path);
      const blob = response.data;
      const contentType = mimeTypeOf(response.headers?.['content-type'], blob.type);
      const kind = previewKindFor(contentType);

      if (kind === 'image' || kind === 'video') {
        const objectUrl = URL.createObjectURL(blob);
        previewObjectUrlRef.current = objectUrl;
        setPreview({
          entry,
          kind,
          objectUrl,
          contentType,
          sizeBytes: blob.size,
        });
        return;
      }

      if (kind === 'text') {
        if (blob.size > TEXT_PREVIEW_MAX_BYTES) {
          setPreview({
            entry,
            kind: 'unsupported',
            contentType,
            sizeBytes: blob.size,
          });
          return;
        }
        setPreview({
          entry,
          kind: 'text',
          text: await blob.text(),
          contentType,
          sizeBytes: blob.size,
        });
        return;
      }

      setPreview({
        entry,
        kind: 'unsupported',
        contentType,
        sizeBytes: blob.size,
      });
    } catch (error) {
      notifyError((await getDownloadErrorMessage(error)) || 'Failed to open file');
      clearPreview();
    } finally {
      setPreviewLoadingPath(null);
    }
  };

  const handleCopy = async (value: string) => {
    if (await copyToClipboard(value)) {
      notifySuccess('Copied path to clipboard');
    } else {
      notifyError('Failed to copy path');
    }
  };

  const handleDelete = async () => {
    if (!filePendingDelete) return;
    setDeleting(true);
    try {
      await floConsoleService.datasourcesService.deleteFile(datasourceId, filePendingDelete.relative_path);
      if (preview?.entry.path === filePendingDelete.path) {
        clearPreview();
      }
      notifySuccess('File deleted');
      setFilePendingDelete(null);
      await loadFiles(pageNumber);
    } catch (error) {
      notifyError(extractErrorMessage(error) || 'Failed to delete file');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="frost-panel ring-frost-border flex min-h-0 flex-col rounded-xl border ring-1">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#EFF0F1] px-6 py-4">
        <div>
          <h2 className="frost-text text-lg font-semibold">Files</h2>
          <p className="frost-text-muted text-sm">Objects under this datasource base path</p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => void loadFiles(pageNumber)}
            disabled={loadingFiles}
            title="Refresh"
          >
            <RefreshCw className={`h-4 w-4 ${loadingFiles ? 'animate-spin' : ''}`} />
          </Button>
          <Button
            size="sm"
            onClick={() => {
              resetUploadForm();
              setUploadDialogOpen(true);
            }}
          >
            <Upload className="mr-1.5 h-4 w-4" />
            Upload
          </Button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-6">
        {loadingFiles && files.length === 0 ? (
          <p className="frost-text-muted text-sm">Loading files…</p>
        ) : files.length === 0 ? (
          <p className="frost-text-muted text-sm">No files yet. Upload one to get started.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Path</TableHead>
                <TableHead className="w-[1%] text-right whitespace-nowrap">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {files.map((entry) => (
                <TableRow key={entry.path}>
                  <TableCell className="frost-text font-medium">
                    <button
                      type="button"
                      className="text-brand text-left hover:underline disabled:opacity-60"
                      title="Preview file"
                      disabled={previewLoadingPath === entry.path}
                      onClick={() => void handlePreview(entry)}
                    >
                      {previewLoadingPath === entry.path ? 'Opening…' : entry.name}
                    </button>
                  </TableCell>
                  <TableCell>
                    <div className="flex min-w-0 items-center gap-1">
                      <span className="frost-text truncate font-mono text-sm" title={entry.path}>
                        {entry.relative_path}
                      </span>
                      <Button
                        variant="ghost"
                        size="sm"
                        title="Copy datasource path"
                        onClick={() => void handleCopy(entry.path)}
                      >
                        <Copy className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        title="Download"
                        loading={downloadingPath === entry.path}
                        disabled={downloadingPath === entry.path || deleting}
                        onClick={() => void handleDownload(entry)}
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        title="Delete"
                        disabled={deleting}
                        onClick={() => setFilePendingDelete(entry)}
                      >
                        <Trash2 className="h-4 w-4 text-red-600" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}

        {(pageNumber > 1 || hasNextPage) && (
          <div className="mt-4 flex items-center justify-end gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={loadingFiles || pageNumber <= 1}
              onClick={() => void loadFiles(pageNumber - 1)}
            >
              Previous
            </Button>
            <span className="frost-text-muted text-sm">Page {pageNumber}</span>
            <Button
              variant="outline"
              size="sm"
              disabled={loadingFiles || !hasNextPage}
              onClick={() => void loadFiles(pageNumber + 1)}
            >
              Next
            </Button>
          </div>
        )}
      </div>

      <Dialog
        open={uploadDialogOpen}
        onOpenChange={(open) => {
          setUploadDialogOpen(open);
          if (!open) resetUploadForm();
        }}
      >
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Upload file</DialogTitle>
            <DialogDescription>
              Store a file in this datasource. Destination path is optional and defaults to the file name.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-4 py-2">
            <div className="flex flex-col gap-2">
              <Label htmlFor="fileStorageUpload">File</Label>
              <Input
                ref={fileInputRef}
                type="file"
                id="fileStorageUpload"
                onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
                disabled={uploading}
                className="frost-control ring-frost-border frost-text file:text-brand w-full cursor-pointer border px-3 py-2 text-sm ring-1 outline-none file:cursor-pointer"
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="fileStorageUploadPath">
                Destination path <span className="frost-text-subtle">(Optional)</span>
              </Label>
              <Input
                id="fileStorageUploadPath"
                placeholder="folder/ or folder/name.ext"
                value={uploadPath}
                onChange={(e) => setUploadPath(e.target.value)}
                disabled={uploading}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setUploadDialogOpen(false)} disabled={uploading}>
              Cancel
            </Button>
            <Button onClick={() => void handleUpload()} loading={uploading} disabled={!file || uploading}>
              Upload
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={preview !== null || previewLoadingPath !== null}
        onOpenChange={(open) => {
          if (!open) clearPreview();
        }}
      >
        <DialogContent className="flex max-h-[90vh] max-w-3xl min-w-0 flex-col overflow-hidden lg:max-w-3xl">
          <DialogHeader>
            <DialogTitle className="truncate pr-6">
              {preview?.entry.name || files.find((entry) => entry.path === previewLoadingPath)?.name || 'File preview'}
            </DialogTitle>
            <DialogDescription className="truncate font-mono text-xs">
              {preview?.entry.relative_path ||
                files.find((entry) => entry.path === previewLoadingPath)?.relative_path ||
                'Loading…'}
            </DialogDescription>
          </DialogHeader>

          <div className="min-h-0 flex-1 overflow-auto py-2">
            {previewLoadingPath && !preview ? (
              <p className="frost-text-muted text-sm">Loading preview…</p>
            ) : preview?.kind === 'image' && preview.objectUrl ? (
              <img
                src={preview.objectUrl}
                alt={preview.entry.name}
                className="mx-auto max-h-[60vh] max-w-full rounded-md object-contain"
              />
            ) : preview?.kind === 'video' && preview.objectUrl ? (
              <video src={preview.objectUrl} controls className="mx-auto max-h-[60vh] max-w-full rounded-md" />
            ) : preview?.kind === 'text' ? (
              <pre className="frost-text bg-muted/40 max-h-[60vh] overflow-auto rounded-md p-4 text-xs break-words whitespace-pre-wrap">
                {preview.text}
              </pre>
            ) : preview ? (
              <div className="frost-text-muted flex flex-col gap-2 text-sm">
                <p>Preview is not available for this file type.</p>
                <p className="font-mono text-xs">
                  {preview.contentType || 'application/octet-stream'} · {preview.sizeBytes} bytes
                </p>
              </div>
            ) : null}
          </div>

          {preview && (
            <DialogFooter>
              <Button variant="outline" onClick={clearPreview}>
                Close
              </Button>
              <Button variant="destructive" disabled={deleting} onClick={() => setFilePendingDelete(preview.entry)}>
                <Trash2 className="mr-1.5 h-3.5 w-3.5" />
                Delete
              </Button>
              <Button onClick={() => void handleDownload(preview.entry)}>
                <Download className="mr-1.5 h-3.5 w-3.5" />
                Download
              </Button>
            </DialogFooter>
          )}
        </DialogContent>
      </Dialog>

      <DeleteConfirmationDialog
        isOpen={filePendingDelete !== null}
        title="Delete file"
        message={
          filePendingDelete
            ? `Delete "${filePendingDelete.name}" (${filePendingDelete.relative_path})? This cannot be undone.`
            : ''
        }
        confirmLabel="Delete"
        loading={deleting}
        onConfirm={handleDelete}
        onCancel={() => {
          if (!deleting) setFilePendingDelete(null);
        }}
      />
    </div>
  );
};

export default FileStoragePanel;
