from src import experiment as ex


def _fake_run(d):
    (d / "final").mkdir(parents=True)
    (d / "final" / "adapter_config.json").write_text("{}")
    (d / "run.json").write_text("{}")


def test_restore_from_drive_copies_when_work_dir_is_empty(tmp_path):
    drive, run_dir = tmp_path / "drive" / "e1", tmp_path / "work" / "e1"
    _fake_run(drive)
    ex._restore_from_drive(run_dir, drive)
    assert (run_dir / "final" / "adapter_config.json").exists() and (run_dir / "run.json").exists()


def test_restore_does_not_overwrite_existing_work_dir(tmp_path):
    drive, run_dir = tmp_path / "drive", tmp_path / "work"
    _fake_run(drive)
    _fake_run(run_dir)
    (run_dir / "run.json").write_text('{"keep": 1}')
    ex._restore_from_drive(run_dir, drive)
    assert (run_dir / "run.json").read_text() == '{"keep": 1}'


def test_restore_noop_when_drive_has_nothing(tmp_path):
    ex._restore_from_drive(tmp_path / "work", tmp_path / "drive")
    assert not (tmp_path / "work").exists()
