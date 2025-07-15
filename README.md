# Fusebuild #

This is a highly experimental beginning of a build tool.

Fusebuild got is name, because it uses a fuse mount to track all file access while building. It is also called fuse build because the aim is to fuse other build systems.

I have earlier worked a lot with Make and Bazel, and got the idea for this from that experience and discussions with colleages.


The overall aims are:
1) Support safe incremental builds to speed up development - like Bazel.
2) Avoid manual dependencies as in Make and Bazel. By letting all file access go through a fuse file system every dependency can be logged and checked for changes at the next build invocation.
3) Don't rewrite the whole world: Focus on providing a thin layer over existing build systems, such Maven and CMAKE etc. can exist in smaller parts. If you have both Java and C++ code, niether Maven, nor CMAKE are good build systems, but they are known solutions for each language. The aim is that Fusebuild can bridge them together, and make sure CMAKE or Maven is called whenever a output from one is used by the other.
To make this practical the option of calling the nested build system incrementally - i.e. no clean before each build invocation - needs to be added.
4) To be simpler than Bazel. For instance, only have actions, not special repository rules. Make it possible to generate actions from output of other actions.
5) Write build in a known language, Python, instead of limiting to Starlark.
6) Provide true hermitic builds by sand-boxing actions into containers.
8) Make it possible to build said containers with Fusebuild itself.

## Design ##

As with Bazel and Make the user write actions. Each action runs on a fuse mount collecting all the file dependencies. At the next build invocation those dependencies are checked, and no rebuild takes place if they are all unchanged.
If an output from another action is accessed, the fuse mount demon stops and builds that action (if needed).
For incremental actions the file access log much be merged for all invocations.
The fuse mount is a mirror of the whole root file system, so any file access, even in the host system is logged.

To make sure that files aren't accessed outside the fuse file system a sandbox is used. Right now there are two sandboxes: NoSandbox, which doesn't provide any thing, because as soon as absolute filenames to the outside is used, all files can be accessed (as in Bazel Linux sandboxes). Then there is a Bwrap sandbox based on Bubblewrap, which simply makes the fuse mount the root filesystem - chroot without being root.

The actions a created by running FUSEBUILD.py files in the source code. The action of creating actions are in themselves actions, running in a Bwrap sandbox.

Notice the Bwrapped fuse mount is actually a traced remount of your real root filesystem, i.e. all tools etc. are from the host at this stage.
Then I am going to make a container based sandbox, where the root filesystem is a container image. I will not use Docker, nor Podman, but simply point to the unpacked layers directly. And make rules for downloading, unpacking and making new layers within Fusebuild. For that I am going to use Rootlesskit.

There is no central deamon running the system as with Bazel and Graddle. All is done with recursive process invocation. You can run as many builds in parallel, and there is a lock around each action making it safe.


## Command line ##

Right now there is a
> fusebuild.sh <catagories\> <actions | directories\>+

to get the list of actions or all actions in the directory(ies) invoked, if they have match the category. A action has the form
> <path to dir with FUSEBUILD.py\>/<name\>.

By pointing to a directory all FUSEBUILD.py below that directory will be checked and all actions with a matching category will be invoked.

> fusebuild.sh build,test .

will be the typical "build and test all" command.



Special categories like "clean" might be added later.


## The future ##

This is a proof of concept. I have no chance to make it even remotely practical: I am only trying it out because I have dreamed this up over many years.
A lot needs to be rewritten in C/C++/Rust/Go to make it faster. Python is way too slow for the fusemount and dependency checking - but perfectly ok to write the build files in, as it is only required to run it or an dependency have changed.
But before that I need to make examples for how to use Maven, CMAKE, plain Make etc. to figure out how much the core needs of extra functionality to actually work as intended. And what the next layer handling toolchains, platforms and targets might be.


