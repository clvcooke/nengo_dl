"""We use semantic versioning (see http://semver.org/).
Additionally, '.dev0' will be added to the version unless the code base
represents a release version. Commits for which the version doesn't have
'.dev0' should be git tagged with the version.
"""

name = "nengo_dl"
version_info = (0, 4, 1)  # (major, minor, patch)
dev = True

version = "{v}{dev}".format(v='.'.join(str(v) for v in version_info),
                            dev='.dev0' if dev else '')
