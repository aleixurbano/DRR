import os
from urllib.parse import unquote, urlparse


def _parse_hf_hub_url(url):
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in {"huggingface.co", "www.huggingface.co"}:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if "resolve" not in parts:
        return None

    resolve_idx = parts.index("resolve")
    if resolve_idx < 1 or resolve_idx + 2 >= len(parts):
        return None

    repo_id = "/".join(parts[:resolve_idx])
    revision = unquote(parts[resolve_idx + 1])
    filename = unquote("/".join(parts[resolve_idx + 2 :]))
    return repo_id, revision, filename


def ensure_huggingface_hub_compat():
    import huggingface_hub

    if hasattr(huggingface_hub, "cached_download"):
        return

    from huggingface_hub import hf_hub_download

    def cached_download(
        url,
        cache_dir=None,
        force_filename=None,
        proxies=None,
        resume_download=None,
        user_agent=None,
        use_auth_token=None,
        local_files_only=False,
        legacy_cache_layout=False,
        library_name=None,
        library_version=None,
        token=None,
        **kwargs,
    ):
        parsed = _parse_hf_hub_url(url)
        resolved_token = use_auth_token if token is None else token

        if parsed is not None:
            repo_id, revision, filename = parsed
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                force_filename=force_filename,
                proxies=proxies,
                resume_download=resume_download,
                user_agent=user_agent,
                token=resolved_token,
                local_files_only=local_files_only,
                library_name=library_name,
                library_version=library_version,
            )

        if force_filename is None:
            force_filename = os.path.basename(urlparse(url).path)
        if cache_dir is None:
            cache_dir = os.getcwd()

        destination = os.path.join(cache_dir, force_filename)
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        if local_files_only:
            if os.path.exists(destination):
                return destination
            raise FileNotFoundError(destination)

        import requests

        headers = {}
        if isinstance(resolved_token, str):
            headers["authorization"] = f"Bearer {resolved_token}"

        response = requests.get(url, stream=True, proxies=proxies, headers=headers, timeout=30)
        response.raise_for_status()

        with open(destination, "wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output_file.write(chunk)

        return destination

    huggingface_hub.cached_download = cached_download
